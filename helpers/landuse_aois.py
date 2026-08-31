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
    # Extension listed in docs/roadmap_segmentation.md §6 as "à évaluer ensuite" for more
    # tiles/diversity -- both port/industrial zones, outside the Nord-Pas-de-Calais cluster
    # so the model sees more than one region's visual signature.
    # NB: Le Havre and Fos-sur-Mer (the towns actually named in the roadmap) sit on a
    # Sentinel-2/MGRS tile boundary -- the windowed read gets silently clipped to a sliver
    # (confirmed empirically: 185x2306px and 2480x60px instead of ~2250x2280). Rouen and
    # Istres are the nearest same-theme substitutes that land fully inside one tile.
    LanduseAOI("rouen", _town_bbox(49.4432, 1.0993), "76"),
    LanduseAOI("istres", _town_bbox(43.5178, 4.9866), "13"),
    # A few more Nord/Pas-de-Calais towns to densify the existing cluster without leaving
    # its MGRS-safe scale.
    LanduseAOI("maubeuge", _town_bbox(50.2775, 3.9714), "59"),
    LanduseAOI("saint_omer", _town_bbox(50.7500, 2.2600), "62"),
    LanduseAOI("arras", _town_bbox(50.2910, 2.7770), "62"),
    # Haute-Savoie: cross-border industrial/commercial zone near Geneva, alpine-valley
    # terrain -- visually very different from the flat northern-France AOIs above.
    LanduseAOI("annemasse", _town_bbox(46.1933, 6.2350), "74"),
    # Targeted at the Parking class specifically -- it's the rarest class by far (~0.2% of
    # pixels dataset-wide) and stayed near-0 IoU even after class weighting, data
    # augmentation and multi-AOI generalization fixes (see docs/roadmap_segmentation.md and
    # the train_unet_landuse.ipynb discussion). None of the AOIs above were chosen for
    # parking density -- these are, centered on large retail/shopping zones known for big
    # surface parking lots (the >1500m^2 loi APER threshold this class targets), spread
    # across regions/departments not yet covered for added visual diversity.
    # Caveat: OSM sometimes maps a mall's parking as a multipolygon *relation* (ring with
    # islands for landscaping/lighting) rather than a plain *way* -- fetch_osm_polygons only
    # reads ways (docs/roadmap_segmentation.md's known limitation, still true), so a share of
    # exactly the biggest parkings these AOIs are chosen for may still be missed. Worth
    # checking the fetch notebook's per-AOI parking way count before assuming this alone
    # fixes the class.
    
    # WARRNING les très grands parkings de centres commerciaux sont souvent cartographiés sur OSM comme des relations 
    # (anneau extérieur + îlots découpés pour les espaces verts/luminaires) plutôt que des way simples. 
    # Ça veut dire qu'une partie — potentiellement les plus gros — des parkings de ces nouvelles AOI pourrait quand même être ratée.
    LanduseAOI("lille_englos", _town_bbox(50.6386, 2.9634), "59"),  # zone commerciale Englos-les-Geants
    LanduseAOI("marseille_plan_de_campagne", _town_bbox(43.4167, 5.3167), "13"),  # one of Europe's largest open-air retail zones
    LanduseAOI("velizy", _town_bbox(48.7833, 2.1917), "78"),  # Velizy 2 shopping center, Paris region
    LanduseAOI("bordeaux_lac", _town_bbox(44.8893, -0.5622), "33"),  # Bordeaux-Lac retail zone
    LanduseAOI("toulouse_labege", _town_bbox(43.5464, 1.5145), "31"),  # Labege retail/technopole zone
]
