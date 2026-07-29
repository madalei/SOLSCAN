from PIL import Image


def crop_to_multiple(image: Image.Image, tile_size: int) -> Image.Image:
    """Center-crop an image so both dimensions are an exact multiple of tile_size."""
    width, height = image.size
    used_width = (width // tile_size) * tile_size
    used_height = (height // tile_size) * tile_size

    left = (width - used_width) // 2
    top = (height - used_height) // 2
    return image.crop((left, top, left + used_width, top + used_height))
