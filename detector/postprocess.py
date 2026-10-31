"""Post-processing for raw YOLO detections: drop duplicate/nested boxes for
the same physical product, and drop boxes that are actually shelf price
tags rather than products. Pure functions, no Flask/env-var reads here (the
service passes in already-resolved config) so this module is also usable
from standalone debug/comparison scripts.
"""
from PIL import Image, ImageDraw


def box_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def dedupe_boxes(boxes, containment_threshold=0.7):
    """Drop boxes that are mostly contained inside a larger box (split
    packs, nested duplicate detections). Standard IoU-based NMS (already
    applied inside YOLO's own predict()) misses this case: a small box
    fully inside a big one can have low IoU (union dominated by the big
    box) while still being a near-total duplicate. This checks containment
    -- intersection over the SMALLER box's own area -- instead.

    Does not merge genuinely separate boxes that only partially overlap
    (e.g. two stacked packs sharing an edge), since their intersection is
    small relative to either box's own area.

    boxes: list of dicts with "bbox" and "score". Returns a filtered list
    (same dicts, original order not preserved -- caller can re-sort).
    """
    ordered = sorted(boxes, key=lambda b: -box_area(b["bbox"]))
    kept = []
    for b in ordered:
        b_area = box_area(b["bbox"])
        is_duplicate = False
        for k in kept:
            inter = intersection_area(b["bbox"], k["bbox"])
            if b_area > 0 and inter / b_area > containment_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(b)
    return kept


def is_price_tag_geometry(bbox, image_w, image_h, max_aspect_ratio, max_height_frac):
    """Price tags are wide-and-short relative to their own shape, and small
    relative to the overall image (a thin label strip, not a product-sized
    box)."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if h <= 0 or image_h <= 0:
        return False
    aspect_ratio = w / h
    height_frac = h / image_h
    return aspect_ratio > max_aspect_ratio and height_frac < max_height_frac


_clip_model = None
_clip_processor = None
_clip_device = None

_PRICE_TAG_PROMPTS = [
    "a price tag label with numbers and currency symbols",
    "a retail product package on a shelf",
]


def _get_clip(model_name, device):
    global _clip_model, _clip_processor, _clip_device
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor

        _clip_device = device
        _clip_processor = CLIPProcessor.from_pretrained(model_name)
        _clip_model = CLIPModel.from_pretrained(model_name).to(device)
        _clip_model.train(False)  # inference mode
    return _clip_model, _clip_processor, _clip_device


def classify_price_tags_clip(crops, model_name, device, threshold):
    """Zero-shot CLIP classification: for each crop, compare against a
    "price tag" prompt vs a "product package" prompt. Returns a list of
    bools, True where the crop looks more like a price tag than the
    threshold allows.

    Loads (and caches) its own CLIP instance on first use -- the detector
    service does not otherwise load CLIP, only the grouping service does.
    """
    import torch

    if not crops:
        return []

    model, processor, device = _get_clip(model_name, device)
    inputs = processor(text=_PRICE_TAG_PROMPTS, images=crops, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)  # [N, 2]
    price_tag_probs = probs[:, 0].cpu().tolist()
    return [p > threshold for p in price_tag_probs]


def draw_debug_boxes(image: Image.Image, boxes: list, color=(255, 0, 0)) -> Image.Image:
    """Simple thin-box + score debug visualization (not the gateway's
    group-colored one -- the detector has no group concept)."""
    vis = image.copy().convert("RGB")
    draw = ImageDraw.Draw(vis)
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{b['score']:.2f}"
        draw.text((x1 + 2, max(0, y1 - 12)), label, fill=color)
    return vis
