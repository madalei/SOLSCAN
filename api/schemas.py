from datetime import date, timedelta

from pydantic import BaseModel, Field, model_validator


class ClassifyRequest(BaseModel):
    bbox: tuple[float, float, float, float] = Field(..., description="min_lon, min_lat, max_lon, max_lat (WGS84)")
    date_start: date = Field(default_factory=lambda: date.today() - timedelta(days=180))
    date_end: date = Field(default_factory=date.today)
    max_cloud_cover: float = Field(default=20.0, ge=0, le=100)

    @model_validator(mode="after")
    def check_bbox(self):
        min_lon, min_lat, max_lon, max_lat = self.bbox
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError("bbox must satisfy min_lon < max_lon and min_lat < max_lat")
        return self


class ClassifyResponse(BaseModel):
    overlay_png_base64: str
    original_png_base64: str
    tile_size: int
    grid_rows: int
    grid_cols: int
    tile_counts: dict[str, int]
    tile_percentages: dict[str, float]
    scene_datetime: str | None
    cloud_cover_pct: float | None
    bbox: tuple[float, float, float, float]
