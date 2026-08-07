"""Fetch OSM (parking/industrial) and Cartofriches (friches) polygons for an AOI.

No API key needed for either source. See docs/roadmap_segmentation.md for the gotchas
this module works around (Overpass User-Agent, Cartofriches BBOX param). Both are free,
shared public instances and occasionally answer with a 502/504 under load -- callers get
retried automatically (mirror rotation for Overpass, backoff for Cartofriches) rather than
failing the whole AOI on a transient hiccup.
"""

import time

import requests

# Mirrors of the same public Overpass instance -- overpass-api.de alone times out often
# enough (504) that rotating through alternates on retry meaningfully improves success rate.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
]
CARTOFRICHES_URL = "https://www.geo2france.fr/geoserver/cerema/ows"
# Overpass 406s on requests' default User-Agent -- needs something explicit.
HEADERS = {"User-Agent": "SOLSCAN/0.1 (educational project, RNCP38616)"}

Ring = list[tuple[float, float]]  # (lon, lat) points


def _close_ring(points: list[tuple[float, float]]) -> Ring:
    if points and points[0] != points[-1]:
        points = points + [points[0]]
    return points


def _request_with_retries(method: str, urls: list[str], *, retry_delay_s: float = 5, **kwargs) -> requests.Response:
    """Try each URL in turn (same request each time), pausing between attempts.

    Raises the last exception if every URL failed -- callers (the fetch AOI loop) already
    catch and skip on failure, so this only needs to make a single AOI's fetch as likely to
    succeed as possible, not guarantee it.
    """
    last_exc: Exception | None = None
    for attempt, url in enumerate(urls):
        try:
            resp = requests.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < len(urls) - 1:
                print(f"  ({url} failed: {exc} -- retrying in {retry_delay_s:.0f}s)")
                time.sleep(retry_delay_s)
    raise last_exc


def fetch_osm_polygons(bbox: tuple[float, float, float, float], timeout: int = 120) -> dict[str, list[Ring]]:
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
    # Client timeout gets extra margin over the server-side [timeout:] budget above, so we're
    # never the ones cutting off a query the server would've otherwise finished.
    resp = _request_with_retries(
        "POST", OVERPASS_URLS, data={"data": query}, headers=HEADERS, timeout=timeout + 30
    )
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
        # Without this the server answers in Lambert-93 (EPSG:2154), which silently breaks
        # _bbox_intersects below (it compares raw coords against a WGS84 bbox) -- every
        # friche then fails the intersection check and gets dropped, with no error raised.
        "srsName": "EPSG:4326",
    }
    # No known mirror for this one (single GeoServer instance) -- just retry the same URL.
    resp = _request_with_retries(
        "GET", [CARTOFRICHES_URL] * 3, params=params, headers=HEADERS, timeout=timeout
    )
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
