from dataclasses import dataclass


@dataclass
class LanduseAOI:
    """One area of interest for the landuse segmentation dataset.

    @param name: short label used as a filename prefix for tiles generated from this AOI
    (avoids collisions between AOIs, see docs/roadmap_segmentation.md §6).
    @param bbox: min_lon, min_lat, max_lon, max_lat (WGS84).
    @param dept_insee_prefix: INSEE department code prefix used to filter Cartofriches'
    `site_id` field via CQL (its WFS BBOX param is unreliable, see docs/roadmap_segmentation.md).
    """

    name: str
    bbox: tuple[float, float, float, float]
    dept_insee_prefix: str


def _town_bbox(lat: float, lon: float, half_lon_deg: float = 0.15, half_lat_deg: float = 0.10) -> tuple[float, float, float, float]:
    """~21km x ~22km box centered on a town (at this latitude), matching the ~475km^2
    AOI size that produced 66 usable tiles in the first (abandoned) attempt."""
    return (lon - half_lon_deg, lat - half_lat_deg, lon + half_lon_deg, lat + half_lat_deg)


# Bassin minier du Nord-Pas-de-Calais (friches denses) + Calais/Boulogne (port/industriel) +
# Dunkerque (reused from the first attempt) -- see docs/roadmap_segmentation.md §6 for the rationale.
# Town-center coordinates are approximate; adjust freely, the fetch pipeline only needs a
# reasonable bbox per AOI, not survey-grade precision.
LANDUSE_AOIS: list[LanduseAOI] = [
    LanduseAOI("dunkerque", _town_bbox(51.0343, 2.3768), "59"),
    LanduseAOI("lens", _town_bbox(50.4322, 2.8329), "62"),
    LanduseAOI("douai", _town_bbox(50.3697, 3.0797), "59"),
    LanduseAOI("valenciennes", _town_bbox(50.3574, 3.5233), "59"),
    LanduseAOI("bethune", _town_bbox(50.5297, 2.6389), "62"),
    LanduseAOI("calais", _town_bbox(50.9513, 1.8587), "62"),
    LanduseAOI("boulogne_sur_mer", _town_bbox(50.7264, 1.6147), "62"),
]
