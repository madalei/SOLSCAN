from PIL import Image, ImageDraw

from helpers.palette import get_class_colors


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
) -> Image.Image:
    """Draw a colored, semi-transparent rectangle per tile on top of the image.

    @param only_classes: if given, tiles whose predicted class isn't in this set are left
    untouched (no rectangle drawn) -- lets callers render a filtered view (e.g. a single class)
    from the same boxes/preds without re-running inference.
    """
    colors = get_class_colors(classes)
    overlay = image.convert("RGBA")
    draw_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(draw_layer)
    for box, pred in zip(boxes, preds):
        label = classes[pred]
        if only_classes is not None and label not in only_classes:
            continue
        color = colors[label] + (alpha,)
        draw.rectangle(box, fill=color, outline=(0, 0, 0, 255))
    return Image.alpha_composite(overlay, draw_layer).convert("RGB")
