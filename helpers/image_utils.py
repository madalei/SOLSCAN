from PIL import Image, ImageDraw, ImageFont

from helpers.palette import get_class_colors

MIN_CONFIDENCE_ALPHA = 40  # keep low-confidence tiles faintly visible instead of near-invisible


def crop_to_multiple(image: Image.Image, tile_size: int) -> Image.Image:
    """Center-crop an image so both dimensions are an exact multiple of tile_size."""
    width, height = image.size
    used_width = (width // tile_size) * tile_size
    used_height = (height // tile_size) * tile_size

    left = (width - used_width) // 2
    top = (height - used_height) // 2
    return image.crop((left, top, left + used_width, top + used_height))


def build_overlay(
    image: Image.Image,
    boxes,
    preds,
    classes: list[str],
    alpha: int = 90,
    only_classes: set[str] | None = None,
    confidences: list[float] | None = None,
) -> Image.Image:
    """Draw a colored, semi-transparent rectangle per tile on top of the image.

    @param only_classes: if given, tiles whose predicted class isn't in this set are left
    untouched (no rectangle drawn) -- lets callers render a filtered view (e.g. a single class)
    from the same boxes/preds without re-running inference.
    @param confidences: if given (one per tile, 0-1), each tile's opacity scales with its
    confidence and the percentage is printed on top of it.
    """
    colors = get_class_colors(classes)
    overlay = image.convert("RGBA")
    draw_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(draw_layer)
    font = ImageFont.load_default()

    for i, (box, pred) in enumerate(zip(boxes, preds)):
        label = classes[pred]
        if only_classes is not None and label not in only_classes:
            continue

        confidence = confidences[i] if confidences is not None else None
        tile_alpha = round(MIN_CONFIDENCE_ALPHA + (alpha - MIN_CONFIDENCE_ALPHA) * confidence) if confidence is not None else alpha
        draw.rectangle(box, fill=colors[label] + (tile_alpha,), outline=(0, 0, 0, 255))

        if confidence is not None:
            text = f"{confidence * 100:.0f}%"
            x0, y0, x1, y1 = box
            text_x0, text_y0, text_x1, text_y1 = draw.textbbox((0, 0), text, font=font)
            text_x = x0 + ((x1 - x0) - (text_x1 - text_x0)) / 2
            text_y = y0 + ((y1 - y0) - (text_y1 - text_y0)) / 2
            draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0, 255))

    return Image.alpha_composite(overlay, draw_layer).convert("RGB")
