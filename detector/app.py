"""Detector microservice.

Loads a YOLOv8 model once at startup and exposes it over a tiny Flask API.
POST /detect  -> bounding boxes for products found in the uploaded image.
GET  /health  -> liveness check.

Between raw YOLO output and the response, a post-processing step (see
postprocess.py) removes duplicate/nested boxes for the same physical
product and drops boxes that are actually shelf price tags rather than
products. Every threshold is env-configurable; response shape is
unchanged either way.
"""
import io
import os
import time
import uuid

import torch
from flask import Flask, jsonify, request
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

import postprocess

app = Flask(__name__)

# --- config (all plug-and-play via env vars, no hardcoded paths) ---
# No default here on purpose: silently falling back to some other model if
# DETECTOR_MODEL isn't set would mean serving predictions from the wrong
# model without anyone noticing (this has already happened once).
MODEL_REPO = os.environ.get("DETECTOR_MODEL")
if not MODEL_REPO:
    raise RuntimeError(
        "DETECTOR_MODEL environment variable is not set. Refusing to start "
        "with an implicit default model. Set DETECTOR_MODEL to a local .pt "
        "path or a HuggingFace repo id (see .env.example)."
    )
MODEL_FILE = os.environ.get("DETECTOR_MODEL_FILE", "best.pt")
CONF_THRESHOLD = float(os.environ.get("DETECTOR_CONF", "0.25"))
IOU_THRESHOLD = float(os.environ.get("DETECTOR_IOU", "0.45"))
IMG_SIZE = int(os.environ.get("DETECTOR_IMGSZ", "960"))
MAX_DET = int(os.environ.get("DETECTOR_MAX_DET", "300"))  # ultralytics' own default

# --- post-processing config ---
# duplicate/nested box removal (issue: same product split into multiple
# overlapping boxes, or one box nested inside another)
DEDUPE_ENABLED = os.environ.get("DETECTOR_DEDUPE_ENABLED", "true").lower() == "true"
CONTAINMENT_IOU = float(os.environ.get("DETECTOR_CONTAINMENT_IOU", "0.7"))

# price-tag filtering: "none" | "geometry" | "clip"
PRICE_TAG_METHOD = os.environ.get("DETECTOR_PRICE_TAG_METHOD", "none").lower()
PRICE_TAG_MAX_ASPECT_RATIO = float(os.environ.get("DETECTOR_PRICE_TAG_MAX_ASPECT_RATIO", "3.0"))
PRICE_TAG_MAX_HEIGHT_FRAC = float(os.environ.get("DETECTOR_PRICE_TAG_MAX_HEIGHT_FRAC", "0.05"))
PRICE_TAG_CLIP_MODEL = os.environ.get("DETECTOR_PRICE_TAG_CLIP_MODEL", "openai/clip-vit-base-patch32")
PRICE_TAG_CLIP_THRESHOLD = float(os.environ.get("DETECTOR_PRICE_TAG_CLIP_THRESHOLD", "0.6"))
CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# debug: save before/after box visualizations for every /detect call
DEBUG_SAVE = os.environ.get("DETECTOR_DEBUG_SAVE", "false").lower() == "true"
DEBUG_DIR = os.environ.get("DETECTOR_DEBUG_DIR", os.path.join(os.path.dirname(__file__), "..", "outputs", "debug"))
if DEBUG_SAVE:
    os.makedirs(DEBUG_DIR, exist_ok=True)


def load_model():
    """Load exactly the weights specified by DETECTOR_MODEL. No implicit
    fallback to a different model on failure -- if this can't be resolved,
    the service must refuse to start rather than silently serving
    predictions from whatever model happens to load instead.

    MODEL_REPO can be a local .pt path, or a HuggingFace Hub repo id (in
    which case DETECTOR_MODEL_FILE names the weights file within that repo).
    """
    if os.path.sep in MODEL_REPO or MODEL_REPO.endswith((".pt", ".pth")):
        if not os.path.isfile(MODEL_REPO):
            raise FileNotFoundError(
                f"DETECTOR_MODEL={MODEL_REPO!r} looks like a local weights path "
                "but the file does not exist. Check STORAGE_ROOT and that the "
                "weights were downloaded (see README)."
            )
        return YOLO(MODEL_REPO)
    weights_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    return YOLO(weights_path)


print(f"[detector] loading model DETECTOR_MODEL={MODEL_REPO!r} (file={MODEL_FILE!r}) ...")
model = load_model()
print(f"[detector] model loaded successfully: {MODEL_REPO}")
print(
    f"[detector] postprocess: dedupe={DEDUPE_ENABLED} (containment_iou={CONTAINMENT_IOU})  "
    f"price_tag_method={PRICE_TAG_METHOD}  debug_save={DEBUG_SAVE}"
)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_REPO})


def filter_price_tags(boxes, image):
    """Returns (kept_boxes, dropped_boxes). method=none passes everything through."""
    if PRICE_TAG_METHOD == "none" or not boxes:
        return boxes, []

    if PRICE_TAG_METHOD == "geometry":
        kept, dropped = [], []
        for b in boxes:
            if postprocess.is_price_tag_geometry(
                b["bbox"], image.width, image.height, PRICE_TAG_MAX_ASPECT_RATIO, PRICE_TAG_MAX_HEIGHT_FRAC
            ):
                dropped.append(b)
            else:
                kept.append(b)
        return kept, dropped

    if PRICE_TAG_METHOD == "clip":
        crops = [image.crop(tuple(b["bbox"])) for b in boxes]
        is_tag = postprocess.classify_price_tags_clip(crops, PRICE_TAG_CLIP_MODEL, CLIP_DEVICE, PRICE_TAG_CLIP_THRESHOLD)
        kept = [b for b, tag in zip(boxes, is_tag) if not tag]
        dropped = [b for b, tag in zip(boxes, is_tag) if tag]
        return kept, dropped

    raise ValueError(f"unknown DETECTOR_PRICE_TAG_METHOD: {PRICE_TAG_METHOD}")


@app.post("/detect")
def detect():
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' file field"}), 400

    file = request.files["image"]
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "uploaded file is not a valid image"}), 400

    start = time.perf_counter()
    results = model.predict(
        source=image,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        max_det=MAX_DET,
        verbose=False,
    )

    raw_boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = float(box.conf[0])
        raw_boxes.append({"bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)], "score": round(score, 4)})

    boxes = raw_boxes
    if DEDUPE_ENABLED:
        boxes = postprocess.dedupe_boxes(boxes, CONTAINMENT_IOU)
    boxes, _dropped_tags = filter_price_tags(boxes, image)

    latency_ms = (time.perf_counter() - start) * 1000

    if DEBUG_SAVE:
        req_id = uuid.uuid4().hex[:10]
        before_name = os.path.splitext(file.filename or "image")[0]
        postprocess.draw_debug_boxes(image, raw_boxes).save(os.path.join(DEBUG_DIR, f"{before_name}_{req_id}_before.jpg"))
        postprocess.draw_debug_boxes(image, boxes).save(os.path.join(DEBUG_DIR, f"{before_name}_{req_id}_after.jpg"))

    return jsonify(
        {
            "boxes": boxes,
            "latency_ms": round(latency_ms, 2),
            "image_size": {"w": image.width, "h": image.height},
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))
