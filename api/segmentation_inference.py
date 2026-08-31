"""U-Net landuse segmentation inference (fond / parking / industriel-commercial / friche).

Mirrors api/inference.py's structure but produces a pixel-level mask (4 classes) instead
of one label per 64px tile (10 EuroSAT classes). NOT wired up to a trained checkpoint yet
-- run notebooks/fetch_landuse_dataset.ipynb then notebooks/train_unet_landuse.ipynb first
(see docs/roadmap_segmentation.md). Untested end-to-end for that reason.
"""

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from api.inference import AOISizeError, AOITooLargeError, AOITooSmallError
from helpers.image_utils import crop_to_multiple
from helpers.mask_postprocess import filter_small_regions
from helpers.mask_rasterize import CLASS_NAMES, CLASS_PARKING, MIN_PARKING_AREA_M2, colorize_mask
from models.unet_builder import build_unet

UNET_CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "unet_landuse.pth"
TILE_SIZE = 256  # matches the training tile size (docs/roadmap_segmentation.md §3)
SENTINEL2_GSD_M = 10  # meters/pixel
MAX_TILES = 100  # 256px tiles cover ~16x more ground area each than the 64px EuroSAT ones


class ModelNotReadyError(Exception):
    """Raised when /v2/classify is called before the U-Net has been trained."""


def load_unet_model(device):
    if not UNET_CHECKPOINT_PATH.exists():
        raise ModelNotReadyError(
            f"No U-Net checkpoint at {UNET_CHECKPOINT_PATH}. Run notebooks/fetch_landuse_dataset.ipynb "
            "then notebooks/train_unet_landuse.ipynb first."
        )
    model = build_unet(num_classes=len(CLASS_NAMES), device=device)
    model.load_state_dict(torch.load(UNET_CHECKPOINT_PATH, map_location=device))
    model.eval()
    return model


def _bbox_dimensions_m(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Rough haversine width/height in meters -- duplicated from api/inference.py (private
    helper there, and small enough not to be worth a cross-module import)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    earth_radius_m = 6_371_000
    lat_mid = math.radians((min_lat + max_lat) / 2)
    width_m = math.radians(max_lon - min_lon) * earth_radius_m * math.cos(lat_mid)
    height_m = math.radians(max_lat - min_lat) * earth_radius_m
    return width_m, height_m


def validate_aoi_bbox(bbox: tuple[float, float, float, float], max_tiles: int = MAX_TILES) -> None:
    """Pre-fetch guard: no network call, just geometry -- raises before anything expensive happens."""
    tile_m = TILE_SIZE * SENTINEL2_GSD_M
    width_m, height_m = _bbox_dimensions_m(bbox)
    if width_m < tile_m or height_m < tile_m:
        raise AOITooSmallError(f"AOI too small: must be at least {tile_m}x{tile_m}m (one landuse tile). Draw a larger rectangle.")
    est_tiles = math.ceil(width_m / tile_m) * math.ceil(height_m / tile_m)
    if est_tiles > max_tiles:
        raise AOITooLargeError(f"AOI too large: ~{est_tiles} tiles estimated (max {max_tiles}). Draw a smaller rectangle.")


def segment_grid(image: Image.Image, model, device, transform, tile_size: int = TILE_SIZE, max_tiles: int = MAX_TILES):
    """Crop to a tile grid, run each tile through the U-Net, stitch predictions into one
    pixel-level mask the size of the (possibly cropped) image.

    Re-validates size post-fetch, same reasoning as api/inference.classify_grid.
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
        preds = model(batch).argmax(1).cpu().numpy()  # (n_tiles, tile_size, tile_size)

    full_mask = np.zeros((image.height, image.width), dtype=np.uint8)
    for (x0, y0, x1, y1), tile_pred in zip(boxes, preds):
        full_mask[y0:y1, x0:x1] = tile_pred

    # Safety net, not the primary filter -- training masks already only label parkings
    # >=1500m^2 (see helpers/mask_rasterize.py), this just catches the rare spurious small
    # blob the model predicts anyway (docs/roadmap_segmentation.md §8).
    full_mask = filter_small_regions(full_mask, CLASS_PARKING, SENTINEL2_GSD_M, MIN_PARKING_AREA_M2)

    return image, full_mask, n_rows, n_cols


def build_segmentation_overlay(image: Image.Image, mask: np.ndarray, alpha: int = 140) -> Image.Image:
    """Colorized mask blended over the image, background (class 0) left fully transparent
    so only parking/industrial/friche pixels are highlighted."""
    colored = colorize_mask(mask).convert("RGBA")
    alpha_channel = np.where(mask == 0, 0, alpha).astype(np.uint8)
    colored.putalpha(Image.fromarray(alpha_channel, mode="L"))
    return Image.alpha_composite(image.convert("RGBA"), colored).convert("RGB")
