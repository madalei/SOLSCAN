"""Turn WGS84 polygon rings (OSM + Cartofriches) into a multi-class raster mask
aligned with a Sentinel-2 scene. See docs/roadmap_segmentation.md §1 and §3.

Class IDs: 0=fond, 1=parking, 2=industriel/commercial, 3=friche.
"""

import numpy as np
import rasterio.features
import rasterio.warp
from PIL import Image

from helpers.geo_fetch import Ring

CLASS_BACKGROUND = 0
CLASS_PARKING = 1
CLASS_INDUSTRIAL = 2
CLASS_FRICHE = 3
CLASS_NAMES = ["Fond", "Parking", "Industriel/commercial", "Friche"]
# For visual inspection only (colorize_mask) -- raw mask files must keep raw class indices.
CLASS_PREVIEW_COLORS = {
    CLASS_BACKGROUND: (30, 30, 30),
    CLASS_PARKING: (255, 215, 0),
    CLASS_INDUSTRIAL: (255, 99, 71),
    CLASS_FRICHE: (148, 0, 211),
}

MIN_PARKING_AREA_M2 = 1500  # proxy for the loi APER (2023) large-parking threshold


def _polygon_area_m2(ring: Ring) -> float:
    """Shoelace formula. `ring` must be closed and in a metric (projected) CRS."""
    area = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _reproject_rings(rings: list[Ring], dst_crs) -> list[Ring]:
    reprojected = []
    for ring in rings:
        geom = rasterio.warp.transform_geom("EPSG:4326", dst_crs, {"type": "Polygon", "coordinates": [ring]})
        reprojected.append(geom["coordinates"][0])
    return reprojected


def rasterize_landuse_mask(
    scene_crs,
    scene_transform,
    scene_shape: tuple[int, int],
    osm_parking: list[Ring],
    osm_industrial: list[Ring],
    cartofriches_friches: list[Ring],
    min_parking_area_m2: float = MIN_PARKING_AREA_M2,
) -> np.ndarray:
    """Burn all polygons into a single uint8 mask, `scene_shape` = (height, width).

    Painted in priority order (later overwrites earlier on overlap): parking, then
    industrial/commercial, then friche last -- a reclaimed industrial site must read as
    friche, per docs/roadmap_segmentation.md §1.
    """
    parking_proj = _reproject_rings(osm_parking, scene_crs)
    industrial_proj = _reproject_rings(osm_industrial, scene_crs)
    friche_proj = _reproject_rings(cartofriches_friches, scene_crs)

    parking_proj = [ring for ring in parking_proj if _polygon_area_m2(ring) >= min_parking_area_m2]

    shapes = []
    for ring in parking_proj:
        shapes.append(({"type": "Polygon", "coordinates": [ring]}, CLASS_PARKING))
    for ring in industrial_proj:
        shapes.append(({"type": "Polygon", "coordinates": [ring]}, CLASS_INDUSTRIAL))
    for ring in friche_proj:
        shapes.append(({"type": "Polygon", "coordinates": [ring]}, CLASS_FRICHE))

    if not shapes:
        return np.zeros(scene_shape, dtype=np.uint8)

    mask = rasterio.features.rasterize(
        shapes,
        out_shape=scene_shape,
        transform=scene_transform,
        fill=CLASS_BACKGROUND,
        dtype=np.uint8,
    )
    return mask


def colorize_mask(mask: np.ndarray) -> Image.Image:
    """RGB visualization of a raw class-index mask (docs/roadmap_segmentation.md §7:
    the raw PNGs aren't visualizable as-is since pixel values are 0..N-1, not 0-255)."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_PREVIEW_COLORS.items():
        rgb[mask == class_id] = color
    return Image.fromarray(rgb, mode="RGB")
