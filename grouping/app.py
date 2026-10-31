"""Grouping microservice.

Given an image and a list of bounding boxes, crops each box, embeds the crops
with a small pretrained CLIP model, and clusters the embeddings into brand
groups with agglomerative clustering (we don't know the number of brands
ahead of time, so a fixed distance threshold works better than fixed k).

POST /group  -> one group id per input box, same order as the input boxes.
GET  /health -> liveness check.
"""
import io
import json
import os
import time

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from transformers import CLIPModel, CLIPProcessor

app = Flask(__name__)

MODEL_NAME = os.environ.get("GROUPING_MODEL", "openai/clip-vit-base-patch32")
DISTANCE_THRESHOLD = float(os.environ.get("GROUPING_DISTANCE_THRESHOLD", "0.3"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[grouping] loading model {MODEL_NAME} on {DEVICE} ...")
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("[grouping] model loaded")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


def embed_crops(crops):
    """Batch all crops through CLIP's image encoder in one forward pass."""
    inputs = processor(images=crops, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)  # L2-normalize
    return features.cpu().numpy()


def cluster_embeddings(embeddings):
    """Agglomerative clustering on cosine distance, cluster count unknown."""
    if len(embeddings) == 1:
        return np.array([0])
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=DISTANCE_THRESHOLD,
    )
    return clustering.fit_predict(embeddings)


def labels_to_group_ids(labels):
    """Map raw cluster labels -> "G1", "G2", ... sorted by descending size."""
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    ordered = sorted(counts, key=lambda label: counts[label], reverse=True)
    label_to_group = {label: f"G{i + 1}" for i, label in enumerate(ordered)}
    return [label_to_group[label] for label in labels]


@app.post("/group")
def group():
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' file field"}), 400
    if "boxes" not in request.form:
        return jsonify({"error": "missing 'boxes' form field (JSON array)"}), 400

    try:
        boxes = json.loads(request.form["boxes"])
    except json.JSONDecodeError:
        return jsonify({"error": "'boxes' is not valid JSON"}), 400

    try:
        image = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
    except Exception:
        return jsonify({"error": "uploaded file is not a valid image"}), 400

    if not boxes:
        return jsonify({"groups": [], "latency_ms": 0.0})

    start = time.perf_counter()

    crops = []
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        crop = image.crop((x1, y1, x2, y2))
        crops.append(crop)

    embeddings = embed_crops(crops)
    labels = cluster_embeddings(embeddings)
    group_ids = labels_to_group_ids(labels)

    latency_ms = (time.perf_counter() - start) * 1000

    return jsonify({"groups": group_ids, "latency_ms": round(latency_ms, 2)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8002)))
