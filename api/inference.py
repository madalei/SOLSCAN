import base64
import io
import math
from pathlib import Path

import numpy as np
import planetary_computer
import pystac_client
import rasterio
import torch
from PIL import Image, ImageDraw
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from helpers.dataloaders import build_eurosat_transform
from helpers.image_utils import crop_to_multiple
from helpers.palette import get_class_colors
from models.resnet18_classifier_builder import build_resnet18_classifier

# Confirmed order: torchvision.datasets.EuroSAT sorts class folders alphabetically (ImageFolder convention)
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "resnet18_eurosat.pth"
TILE_SIZE = 64
SENTINEL2_GSD_M = 10  # meters/pixel
MAX_TILES = 500  # keeps the synchronous /classify request bounded


class NoSceneFoundError(Exception):
    pass


class AOISizeError(Exception):
    """Base class for AOI-size validation failures -- caught together by a single FastAPI exception handler."""


class AOITooSmallError(AOISizeError):
    pass


class AOITooLargeError(AOISizeError):
    pass


def load_model(device):
    model = build_resnet18_classifier(num_classes=len(EUROSAT_CLASSES), device=device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    return model


def _bbox_dimensions_m(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Rough haversine width/height in meters -- no network call, no geo dependency needed."""
    min_lon, min_lat, max_lon, max_lat = bbox
    earth_radius_m = 6_371_000
    lat_mid = math.radians((min_lat + max_lat) / 2)
    width_m = math.radians(max_lon - min_lon) * earth_radius_m * math.cos(lat_mid)
    height_m = math.radians(max_lat - min_lat) * earth_radius_m
    return width_m, height_m


def estimate_tile_count(bbox: tuple[float, float, float, float]) -> int:
    """Upper-bound AOI-size guard (uses ceil, so it over-estimates -- fine for a 'too large' check)."""
    width_m, height_m = _bbox_dimensions_m(bbox)
    tile_m = TILE_SIZE * SENTINEL2_GSD_M
    return math.ceil(width_m / tile_m) * math.ceil(height_m / tile_m)


def is_aoi_too_small(bbox: tuple[float, float, float, float]) -> bool:
    """crop_to_multiple floor-divides, so an AOI under one tile's footprint in either dimension yields 0 tiles."""
    width_m, height_m = _bbox_dimensions_m(bbox)
    tile_m = TILE_SIZE * SENTINEL2_GSD_M
    return width_m < tile_m or height_m < tile_m


def validate_aoi_bbox(bbox: tuple[float, float, float, float], max_tiles: int = MAX_TILES) -> None:
    """Pre-fetch guard: no network call, just geometry -- raises before anything expensive happens."""
    tile_m = TILE_SIZE * SENTINEL2_GSD_M
    if is_aoi_too_small(bbox):
        raise AOITooSmallError(f"AOI too small: must be at least {tile_m}x{tile_m}m (one EuroSAT tile). Draw a larger rectangle.")
    est_tiles = estimate_tile_count(bbox)
    if est_tiles > max_tiles:
        raise AOITooLargeError(f"AOI too large: ~{est_tiles} tiles estimated (max {max_tiles}). Draw a smaller rectangle.")


def fetch_sentinel2_rgb(bbox, date_start: str, date_end: str, max_cloud_cover: float) -> tuple[Image.Image, dict]:
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{date_start}/{date_end}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    items = sorted(search.item_collection(), key=lambda it: it.properties["eo:cloud_cover"])
    if not items:
        raise NoSceneFoundError(
            f"No Sentinel-2 scene found for this AOI between {date_start} and {date_end} "
            f"with cloud cover < {max_cloud_cover}%. Try widening the date range or raising max_cloud_cover."
        )
    item = items[0]

    arrays = []
    for band in ("B04", "B03", "B02"):
        with rasterio.open(item.assets[band].href) as src:
            bounds = transform_bounds("EPSG:4326", src.crs, *bbox)
            window = from_bounds(*bounds, transform=src.transform)
            arrays.append(src.read(1, window=window))

    rgb_raw = np.stack(arrays, axis=-1)
    # Sentinel-2 L2A reflectance is scaled 0-10000; stretch to 8-bit for a viewable/trainable image
    rgb_uint8 = np.clip(rgb_raw / 3000 * 255, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb_uint8, mode="RGB")

    metadata = {
        "scene_datetime": item.properties.get("datetime"),
        "cloud_cover_pct": item.properties.get("eo:cloud_cover"),
    }
    return image, metadata


def classify_grid(image: Image.Image, model, device, transform, tile_size: int = TILE_SIZE, max_tiles: int = MAX_TILES):
    """Crop to a tile grid, classify each tile, return the (possibly cropped) image alongside results.

    Re-validates size post-fetch (defense in depth: the actual STAC raster can come back smaller/larger
    than estimated near scene edges, even though validate_aoi_bbox already checked the requested bbox).
    """
    image = crop_to_multiple(image, tile_size)
    n_cols = image.width // tile_size
    n_rows = image.height // tile_size

    if n_rows == 0 or n_cols == 0:
        raise AOITooSmallError(f"AOI too small after fetch: crop yielded {n_rows}x{n_cols} tiles. Draw a larger rectangle.")
    if n_rows * n_cols > max_tiles:
        raise AOITooLargeError(f"AOI produced {n_rows * n_cols} tiles after fetch (max {max_tiles}). Draw a smaller rectangle.")

    tiles, boxes = [], []
    for row in range(n_rows):
        for col in range(n_cols):
            box = (col * tile_size, row * tile_size, (col + 1) * tile_size, (row + 1) * tile_size)
            tiles.append(transform(image.crop(box)))
            boxes.append(box)

    batch = torch.stack(tiles).to(device)
    with torch.no_grad():
        preds = model(batch).argmax(1).cpu().tolist()

    return image, preds, boxes, n_rows, n_cols


def build_overlay(image: Image.Image, boxes, preds, classes: list[str], alpha: int = 90) -> Image.Image:
    colors = get_class_colors(classes)
    overlay = image.convert("RGBA")
    draw_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(draw_layer)
    for box, pred in zip(boxes, preds):
        color = colors[classes[pred]] + (alpha,)
        draw.rectangle(box, fill=color, outline=(0, 0, 0, 255))
    return Image.alpha_composite(overlay, draw_layer).convert("RGB")


def image_to_base64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
