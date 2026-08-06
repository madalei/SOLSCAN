from collections import Counter
from contextlib import asynccontextmanager

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
from api.schemas import ClassifyRequest, ClassifyResponse
from helpers.dataloaders import build_eurosat_transform


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    app.state.device = device
    app.state.model = load_model(device)
    app.state.transform = build_eurosat_transform(TILE_SIZE)
    yield


app = FastAPI(title="SOLSCAN classifier API", lifespan=lifespan)


@app.exception_handler(AOISizeError)
async def aoi_size_error_handler(_request: Request, exc: AOISizeError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(NoSceneFoundError)
async def no_scene_error_handler(_request: Request, exc: NoSceneFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


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
    image, preds, boxes, n_rows, n_cols = classify_grid(image, app.state.model, app.state.device, app.state.transform)
    overlay = build_overlay(image, boxes, preds, EUROSAT_CLASSES)

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
        scene_datetime=scene_meta["scene_datetime"],
        cloud_cover_pct=scene_meta["cloud_cover_pct"],
        bbox=request.bbox,
    )
