"""Gateway microservice.

The only service the browser talks to. Accepts an uploaded shelf image,
forwards it to the detector then the grouping service (over plain HTTP --
service URLs come from env vars, never hardcoded paths), draws a color-coded
visualization, saves it to outputs/, and returns the combined JSON result.
"""
import io
import json
import os
import time
import uuid

import requests
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image

from visualize import color_for_group, draw

app = Flask(__name__)

DETECTOR_URL = os.environ.get("DETECTOR_URL", "http://localhost:8001")
GROUPING_URL = os.environ.get("GROUPING_URL", "http://localhost:8002")
REQUEST_TIMEOUT = float(os.environ.get("SERVICE_TIMEOUT", "60"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "..", "outputs"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "20")) * 1024 * 1024

os.makedirs(OUTPUT_DIR, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/outputs/<path:filename>")
def get_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.post("/api/analyze")
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' file field"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    image_bytes = file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "uploaded file is not a valid image"}), 400

    request_id = uuid.uuid4().hex[:12]
    total_start = time.perf_counter()

    # --- stage 1: detection ---
    try:
        detect_resp = requests.post(
            f"{DETECTOR_URL}/detect",
            files={"image": (file.filename, image_bytes, file.mimetype)},
            timeout=REQUEST_TIMEOUT,
        )
        detect_resp.raise_for_status()
    except requests.exceptions.Timeout:
        return jsonify({"error": "detector service timed out"}), 504
    except requests.exceptions.RequestException as exc:
        return jsonify({"error": f"detector service unavailable: {exc}"}), 502

    detect_data = detect_resp.json()
    boxes = detect_data["boxes"]

    # --- stage 2: grouping ---
    group_ids = []
    group_latency_ms = 0.0
    if boxes:
        try:
            group_resp = requests.post(
                f"{GROUPING_URL}/group",
                files={"image": (file.filename, image_bytes, file.mimetype)},
                data={"boxes": json.dumps(boxes)},
                timeout=REQUEST_TIMEOUT,
            )
            group_resp.raise_for_status()
        except requests.exceptions.Timeout:
            return jsonify({"error": "grouping service timed out"}), 504
        except requests.exceptions.RequestException as exc:
            return jsonify({"error": f"grouping service unavailable: {exc}"}), 502

        group_data = group_resp.json()
        group_ids = group_data["groups"]
        group_latency_ms = group_data["latency_ms"]

    # --- assemble products + groups ---
    products = []
    for i, (box, group_id) in enumerate(zip(boxes, group_ids)):
        products.append(
            {
                "id": f"P{i + 1}",
                "bbox": box["bbox"],
                "score": box["score"],
                "group_id": group_id,
            }
        )

    group_counts = {}
    for p in products:
        group_counts[p["group_id"]] = group_counts.get(p["group_id"], 0) + 1
    group_order = sorted(group_counts, key=lambda gid: group_counts[gid], reverse=True)
    groups = [
        {
            "group_id": gid,
            "color": "#%02x%02x%02x" % color_for_group(i),
            "count": group_counts[gid],
        }
        for i, gid in enumerate(group_order)
    ]

    # --- visualization ---
    vis_image = draw(image, products, group_order)
    vis_filename = f"{request_id}.jpg"
    vis_image.save(os.path.join(OUTPUT_DIR, vis_filename), "JPEG", quality=90)

    total_latency_ms = (time.perf_counter() - total_start) * 1000

    return jsonify(
        {
            "request_id": request_id,
            "image_size": detect_data["image_size"],
            "num_products": len(products),
            "num_groups": len(groups),
            "products": products,
            "groups": groups,
            "latency_ms": {
                "detect": detect_data["latency_ms"],
                "group": group_latency_ms,
                "total": round(total_latency_ms, 2),
            },
            "visualization_url": f"/outputs/{vis_filename}",
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
