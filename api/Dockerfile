# Build context is the repo root (docker-compose sets this); needs helpers/ and models/ as siblings of api/.
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /usr/local/bin/

# rasterio's wheel bundles GDAL/PROJ/GEOS but still needs libexpat from the system
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"

# Install dependencies first (separate layer -- cached unless pyproject.toml/uv.lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Only what api/main.py's import chain actually needs at runtime
COPY helpers/ helpers/
COPY models/ models/
COPY checkpoints/resnet18_eurosat.pth checkpoints/resnet18_eurosat.pth
COPY checkpoints/resnet18_eurosat_label_smoothing.pth checkpoints/resnet18_eurosat_label_smoothing.pth
# NB: fails the build until this exists -- train it first (notebooks/fetch_landuse_dataset.ipynb
# then notebooks/train_unet_landuse.ipynb). Added ahead of time on request.
COPY checkpoints/unet_landuse.pth checkpoints/unet_landuse.pth
COPY api/ api/
RUN uv sync --frozen --no-dev

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
