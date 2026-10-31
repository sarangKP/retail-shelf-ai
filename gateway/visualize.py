"""Draws color-coded bounding boxes per brand group on the original image."""
import colorsys

from PIL import Image, ImageDraw, ImageFont

FONT = ImageFont.load_default()


def color_for_group(index):
    """Deterministic, visually distinct color per group index (HSV wheel)."""
    hue = (index * 0.6180339887) % 1.0  # golden ratio spacing -> spread-out hues
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def draw(image: Image.Image, products: list[dict], group_order: list[str]) -> Image.Image:
    """products: [{bbox, group_id}, ...]; group_order: group ids sorted by size.

    Returns a new image with boxes + group-id labels drawn on it.
    """
    vis = image.copy().convert("RGB")
    draw_ctx = ImageDraw.Draw(vis)
    group_color = {gid: color_for_group(i) for i, gid in enumerate(group_order)}

    for product in products:
        x1, y1, x2, y2 = product["bbox"]
        color = group_color.get(product["group_id"], (255, 255, 255))
        draw_ctx.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = product["group_id"]
        text_bbox = draw_ctx.textbbox((0, 0), label, font=FONT)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        draw_ctx.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=color)
        draw_ctx.text((x1 + 2, y1 - text_h - 2), label, fill=(0, 0, 0), font=FONT)

    return vis
