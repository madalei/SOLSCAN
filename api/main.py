from collections import Counter
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from datetime import date, timedelta

from api.inference import (
    EUROSAT_CLASSES,
    TILE_SIZE,
    AOISizeError,
    NoSceneFoundError,
    build_overlay,
    classify_grid,
    fetch_sentinel2_rgb,
    image_to_base64_png,
    load_model,
    validate_aoi_bbox,
)
from api.schemas import ClassifyRequest, ClassifyResponse, SegmentResponse
from api.segmentation_inference import (
    ModelNotReadyError,
    build_segmentation_overlay,
    load_unet_model,
    segment_grid,
)
from api.segmentation_inference import TILE_SIZE as UNET_TILE_SIZE
from api.segmentation_inference import validate_aoi_bbox as validate_aoi_bbox_v2
from helpers.dataloaders import build_eurosat_transform
from helpers.mask_rasterize import CLASS_NAMES as LANDUSE_CLASSES
from helpers.segmentation_dataset import build_segmentation_image_transform


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    app.state.device = device
    app.state.model = load_model(device)
    app.state.transform = build_eurosat_transform(TILE_SIZE)
    # U-Net isn't trained yet (see docs/roadmap_segmentation.md) -- loaded lazily on first
    # /v2/classify call instead of at startup, so a missing checkpoint doesn't break the whole API.
    app.state.unet_model = None
    app.state.unet_transform = build_segmentation_image_transform()
    yield


app = FastAPI(title="SOLSCAN classifier API", lifespan=lifespan)


@app.exception_handler(AOISizeError)
async def aoi_size_error_handler(_request: Request, exc: AOISizeError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(NoSceneFoundError)
async def no_scene_error_handler(_request: Request, exc: NoSceneFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ModelNotReadyError)
async def model_not_ready_handler(_request: Request, exc: ModelNotReadyError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/classes")
def get_classes():
    return {"classes": EUROSAT_CLASSES}


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    validate_aoi_bbox(request.bbox)

    DEFAULT_DATE_END = date.today()
    DEFAULT_DATE_START = DEFAULT_DATE_END - timedelta(days=180)
    DEFAULT_MAX_CLOUD_COVER = 10

    image, scene_meta = fetch_sentinel2_rgb(
        bbox=request.bbox,
        date_start=str(DEFAULT_DATE_START),
        date_end=str(DEFAULT_DATE_END),
        max_cloud_cover=DEFAULT_MAX_CLOUD_COVER,
    )
    image, preds, confidences, boxes, n_rows, n_cols = classify_grid(image, app.state.model, app.state.device, app.state.transform)
    overlay = build_overlay(image, boxes, preds, EUROSAT_CLASSES, confidences=confidences)

    counts = Counter(EUROSAT_CLASSES[p] for p in preds)
    total = len(preds)
    tile_counts = {c: counts.get(c, 0) for c in EUROSAT_CLASSES}
    tile_percentages = {c: round(counts.get(c, 0) / total * 100, 2) for c in EUROSAT_CLASSES}

    return ClassifyResponse(
        overlay_png_base64=image_to_base64_png(overlay),
        original_png_base64=image_to_base64_png(image),
        tile_size=TILE_SIZE,
        grid_rows=n_rows,
        grid_cols=n_cols,
        tile_counts=tile_counts,
        tile_percentages=tile_percentages,
        tile_labels=[EUROSAT_CLASSES[p] for p in preds],
        tile_confidences=confidences,
        scene_datetime=scene_meta["scene_datetime"],
        cloud_cover_pct=scene_meta["cloud_cover_pct"],
        bbox=request.bbox,
    )


@app.post("/v2/classify", response_model=SegmentResponse)
def classify_v2(request: ClassifyRequest):
    """Pixel-level landuse segmentation (U-Net) -- see api/segmentation_inference.py.

    Not tested end-to-end yet: no checkpoint exists until notebooks/train_unet_landuse.ipynb
    has been run (see docs/roadmap_segmentation.md). Returns 503 until then.
    """
    validate_aoi_bbox_v2(request.bbox)

    if app.state.unet_model is None:
        app.state.unet_model = load_unet_model(app.state.device)

    DEFAULT_DATE_END = date.today()
    DEFAULT_DATE_START = DEFAULT_DATE_END - timedelta(days=180)
    DEFAULT_MAX_CLOUD_COVER = 10

    image, scene_meta = fetch_sentinel2_rgb(
        bbox=request.bbox,
        date_start=str(DEFAULT_DATE_START),
        date_end=str(DEFAULT_DATE_END),
        max_cloud_cover=DEFAULT_MAX_CLOUD_COVER,
    )
    image, mask, n_rows, n_cols = segment_grid(image, app.state.unet_model, app.state.device, app.state.unet_transform)
    overlay = build_segmentation_overlay(image, mask)

    class_ids, counts = np.unique(mask, return_counts=True)
    total_px = int(mask.size)
    class_pixel_counts = {name: 0 for name in LANDUSE_CLASSES}
    for class_id, count in zip(class_ids.tolist(), counts.tolist()):
        class_pixel_counts[LANDUSE_CLASSES[class_id]] = count
    class_pixel_percentages = {name: round(100 * count / total_px, 2) for name, count in class_pixel_counts.items()}

    return SegmentResponse(
        overlay_png_base64=image_to_base64_png(overlay),
        original_png_base64=image_to_base64_png(image),
        tile_size=UNET_TILE_SIZE,
        grid_rows=n_rows,
        grid_cols=n_cols,
        class_pixel_counts=class_pixel_counts,
        class_pixel_percentages=class_pixel_percentages,
        scene_datetime=scene_meta["scene_datetime"],
        cloud_cover_pct=scene_meta["cloud_cover_pct"],
        bbox=request.bbox,
    )
