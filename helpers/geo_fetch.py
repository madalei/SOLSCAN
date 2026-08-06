"""Fetch OSM (parking/industrial) and Cartofriches (friches) polygons for an AOI.

No API key needed for either source. See docs/roadmap_segmentation.md for the gotchas
this module works around (Overpass User-Agent, Cartofriches BBOX param).
"""

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
CARTOFRICHES_URL = "https://www.geo2france.fr/geoserver/cerema/ows"
# Overpass 406s on requests' default User-Agent -- needs something explicit.
HEADERS = {"User-Agent": "SOLSCAN/0.1 (educational project, RNCP38616)"}

Ring = list[tuple[float, float]]  # (lon, lat) points


def _close_ring(points: list[tuple[float, float]]) -> Ring:
    if points and points[0] != points[-1]:
        points = points + [points[0]]
    return points


def fetch_osm_polygons(bbox: tuple[float, float, float, float], timeout: int = 90) -> dict[str, list[Ring]]:
    """Fetch parking and industrial/commercial way polygons from OSM (Overpass API).

    @param bbox: min_lon, min_lat, max_lon, max_lat (WGS84).
    @return {"parking": [...rings...], "industrial": [...rings...]}, each ring a closed
    list of (lon, lat) points. Only closed ways are usable as polygons -- relations
    (multipolygons) are skipped (see docs/roadmap_segmentation.md, not handled yet).
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    overpass_bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"  # Overpass wants south,west,north,east

    query = f"""
    [out:json][timeout:{timeout}];
    (
      way["amenity"="parking"]({overpass_bbox});
      way["building"~"^(industrial|warehouse)$"]({overpass_bbox});
      way["landuse"~"^(industrial|commercial)$"]({overpass_bbox});
    );
    out geom;
    """
    resp = requests.post(OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    elements = resp.json().get("elements", [])

    polygons: dict[str, list[Ring]] = {"parking": [], "industrial": []}
    seen_ids = set()
    for el in elements:
        if el.get("type") != "way" or el["id"] in seen_ids:
            continue
        seen_ids.add(el["id"])

        geometry = el.get("geometry")
        if not geometry or len(geometry) < 3:
            continue  # not a usable polygon (open way / missing nodes)

        ring = _close_ring([(pt["lon"], pt["lat"]) for pt in geometry])
        tags = el.get("tags", {})
        if tags.get("amenity") == "parking":
            polygons["parking"].append(ring)
        elif tags.get("building") in ("industrial", "warehouse") or tags.get("landuse") in ("industrial", "commercial"):
            polygons["industrial"].append(ring)

    return polygons


def _bbox_intersects(ring: Ring, bbox: tuple[float, float, float, float]) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    return max(lons) >= min_lon and min(lons) <= max_lon and max(lats) >= min_lat and min(lats) <= max_lat


def fetch_cartofriches_polygons(dept_insee_prefix: str, bbox: tuple[float, float, float, float], timeout: int = 90) -> list[Ring]:
    """Fetch friche (brownfield) polygons for a whole department, then keep only those
    whose bbox intersects the AOI (client-side filter -- the WFS BBOX param is unreliable
    on this GeoServer, see docs/roadmap_segmentation.md).

    @param dept_insee_prefix: e.g. "59" (Nord), "62" (Pas-de-Calais) -- filters Cartofriches'
    `site_id` field, which is prefixed by the INSEE commune code.
    """
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": "cerema:cartofriche",
        "outputFormat": "application/json",
        "CQL_FILTER": f"site_id LIKE '{dept_insee_prefix}%'",
    }
    resp = requests.get(CARTOFRICHES_URL, params=params, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    features = resp.json().get("features", [])

    rings: list[Ring] = []
    for feature in features:
        geom = feature.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if not coords:
            continue

        candidate_rings = [coords[0]] if gtype == "Polygon" else [poly[0] for poly in coords] if gtype == "MultiPolygon" else []
        for raw_ring in candidate_rings:
            ring = _close_ring([(pt[0], pt[1]) for pt in raw_ring])
            if _bbox_intersects(ring, bbox):
                rings.append(ring)

    return rings
