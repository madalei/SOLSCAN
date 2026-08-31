from dataclasses import dataclass


@dataclass
class LanduseAOI:
    """One area of interest for the landuse segmentation dataset.

    @param name: short label used as a filename prefix for tiles generated from this AOI
    (avoids collisions between AOIs, see docs/roadmap_segmentation.md §6).
    @param bbox: min_lon, min_lat, max_lon, max_lat (WGS84).
    """

    name: str
    bbox: tuple[float, float, float, float]


def _town_bbox(lat: float, lon: float, half_lon_deg: float = 0.15, half_lat_deg: float = 0.10) -> tuple[float, float, float, float]:
    """~21km x ~22km box centered on a town (at this latitude), matching the ~475km^2
    AOI size that produced 66 usable tiles in the first (abandoned) attempt."""
    return (lon - half_lon_deg, lat - half_lat_deg, lon + half_lon_deg, lat + half_lat_deg)


# Bassin minier du Nord-Pas-de-Calais (friches denses) + Calais/Boulogne (port/industriel) +
# Dunkerque (reused from the first attempt) -- see docs/roadmap_segmentation.md §6 for the rationale.
# Town-center coordinates are approximate; adjust freely, the fetch pipeline only needs a
# reasonable bbox per AOI, not survey-grade precision.
LANDUSE_AOIS: list[LanduseAOI] = [
    LanduseAOI("dunkerque", _town_bbox(51.0343, 2.3768)),
    LanduseAOI("lens", _town_bbox(50.4322, 2.8329)),
    LanduseAOI("douai", _town_bbox(50.3697, 3.0797)),
    LanduseAOI("valenciennes", _town_bbox(50.3574, 3.5233)),
    LanduseAOI("bethune", _town_bbox(50.5297, 2.6389)),
    LanduseAOI("calais", _town_bbox(50.9513, 1.8587)),
    LanduseAOI("boulogne_sur_mer", _town_bbox(50.7264, 1.6147)),
    # Extension listed in docs/roadmap_segmentation.md §6 as "à évaluer ensuite" for more
    # tiles/diversity -- both port/industrial zones, outside the Nord-Pas-de-Calais cluster
    # so the model sees more than one region's visual signature.
    # NB: Le Havre and Fos-sur-Mer (the towns actually named in the roadmap) sit on a
    # Sentinel-2/MGRS tile boundary -- the windowed read gets silently clipped to a sliver
    # (confirmed empirically: 185x2306px and 2480x60px instead of ~2250x2280). Rouen and
    # Istres are the nearest same-theme substitutes that land fully inside one tile.
    LanduseAOI("rouen", _town_bbox(49.4432, 1.0993)),
    LanduseAOI("istres", _town_bbox(43.5178, 4.9866)),
    # A few more Nord/Pas-de-Calais towns to densify the existing cluster without leaving
    # its MGRS-safe scale.
    LanduseAOI("maubeuge", _town_bbox(50.2775, 3.9714)),
    LanduseAOI("saint_omer", _town_bbox(50.7500, 2.2600)),
    LanduseAOI("arras", _town_bbox(50.2910, 2.7770)),
    # Haute-Savoie: cross-border industrial/commercial zone near Geneva, alpine-valley
    # terrain -- visually very different from the flat northern-France AOIs above.
    LanduseAOI("annemasse", _town_bbox(46.1933, 6.2350)),
    # Targeted at the Parking class specifically -- it's the rarest class by far (~0.2% of
    # pixels dataset-wide) and stayed near-0 IoU even after class weighting, data
    # augmentation and multi-AOI generalization fixes (see docs/roadmap_segmentation.md and
    # the train_unet_landuse.ipynb discussion). None of the AOIs above were chosen for
    # parking density -- these are, centered on large retail/shopping zones known for big
    # surface parking lots (the >1500m^2 loi APER threshold this class targets), spread
    # across regions/departments not yet covered for added visual diversity.
    #
    # Caveat: OSM sometimes maps a mall's parking as a multipolygon *relation* (ring with
    # islands for landscaping/lighting) rather than a plain *way* -- fetch_osm_polygons
    # handles this for parking now (see helpers/geo_fetch.py), but a boundary fragmented
    # across several "outer" member ways is still an approximation (no shapely dependency).
    #
    # NB: an earlier version of this list had lille_englos (50.6386, 2.9634), marseille
    # Plan-de-Campagne (43.4167, 5.3167), velizy (48.7833, 2.1917) and bordeaux_lac
    # (44.8893, -0.5622) here -- all four silently read back a sliver instead of the full
    # AOI (same MGRS-tile-boundary issue as Le Havre/Fos-sur-Mer above; confirmed by
    # directly probing rasterio's actual read shape against Planetary Computer, e.g.
    # lille_englos read 152x806px of a nominal ~2226x2126px window). lille_englos had no
    # working substitute nearby -- the whole Lille conurbation sits on the same tile edge,
    # and douai/lens above already cover that tile's usable portion -- so it was dropped.
    # The other three were replaced below with the nearest verified-clean equivalent
    # (ratio of actual-to-nominal read pixels ~1.0, checked directly, not assumed).
    LanduseAOI("vitrolles", _town_bbox(43.4581, 5.2483)),  # retail zone next to Plan-de-Campagne, different MGRS tile
    LanduseAOI("val_europe", _town_bbox(48.8720, 2.7850)),  # Disneyland Paris / Val d'Europe -- one of the largest parking areas in the Paris region
    LanduseAOI("bordeaux_merignac", _town_bbox(44.8407, -0.6478)),  # Merignac retail zone, next to Bordeaux airport
    LanduseAOI("toulouse_labege", _town_bbox(43.5464, 1.5145)),  # Labege retail/technopole zone
]
