"""
Récupère les polygones géographiques qui servent de masques d'entraînement, depuis deux sources gratuites et sans clé API :

Fetch OSM (parking/industrial) and Cartofriches (friches) polygons for an AOI.

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


def _relation_member_rings(el: dict) -> tuple[list[Ring], list[Ring]]:
    """Extract (outer_rings, inner_rings) from an Overpass multipolygon *relation* element.

    `out geom;` embeds each member way's point geometry directly in the relation response --
    no separate recursed way query needed. Inner rings (role="inner") are holes (typically
    small landscaping/lighting islands inside a large parking lot); callers should paint
    them back to background rather than skip them outright, since skipping would just as
    wrongly paint the hole area as the outer class.

    Simplification: each "outer"-role member way is treated as its own closed ring. This is
    correct for the common case (one outer way = the whole boundary) but an approximation
    for a boundary split across several outer ways -- not handled, no shapely/geopandas
    dependency in this project (see docs/roadmap_segmentation.md). Still strictly better
    than skipping relations entirely (the previous behavior).
    """
    outer_rings, inner_rings = [], []
    for member in el.get("members", []):
        if member.get("type") != "way":
            continue
        geometry = member.get("geometry")
        if not geometry or len(geometry) < 3:
            continue
        ring = _close_ring([(pt["lon"], pt["lat"]) for pt in geometry])
        if member.get("role") == "inner":
            inner_rings.append(ring)
        else:  # "outer" or unspecified -- treat as outer
            outer_rings.append(ring)
    return outer_rings, inner_rings


def fetch_osm_polygons(bbox: tuple[float, float, float, float], timeout: int = 120) -> dict[str, list[Ring]]:
    """Fetch parking, industrial/commercial and residential polygons from OSM (Overpass API),
    from plain ways plus (parking only) multipolygon relations.

    @param bbox: min_lon, min_lat, max_lon, max_lat (WGS84).
    @return {"parking": [...], "industrial": [...], "residential": [...], "holes": [...]},
    each a list of closed (lon, lat) rings. "holes" collects the inner rings of parking
    multipolygon relations -- they don't need their own class, painting them as background
    is enough (see `rasterize_landuse_mask`'s `osm_holes` param).

    Large parking lots (the >1500m^2 loi APER threshold this project targets) are often
    mapped on OSM as a multipolygon relation -- an outer ring plus inner-ring holes cut out
    for landscaping/lighting islands -- rather than a plain way; ways alone under-count
    exactly the biggest parkings this project cares about.

    Relations are fetched for `amenity=parking` only, not industrial/residential: a first
    version queried all three and every AOI started timing out on every Overpass mirror,
    even ones that used to fetch fine way-only -- a `landuse=residential` relation can span a
    whole neighborhood with hundreds of member ways, making `out geom;` resolve it far more
    expensive server-side. Parking is also the only class this was meant to fix (see
    docs/roadmap_segmentation.md and the train_unet_landuse.ipynb discussion) -- ways already
    cover industrial/residential reasonably.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    overpass_bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"  # Overpass wants south,west,north,east

    query = f"""
    [out:json][timeout:{timeout}];
    (
      way["amenity"="parking"]({overpass_bbox});
      way["building"~"^(industrial|warehouse)$"]({overpass_bbox});
      way["landuse"~"^(industrial|commercial)$"]({overpass_bbox});
      way["landuse"="residential"]({overpass_bbox});
      rel["amenity"="parking"]["type"="multipolygon"]({overpass_bbox});
    );
    out geom;
    """
    # Client timeout gets extra margin over the server-side [timeout:] budget above, so we're
    # never the ones cutting off a query the server would've otherwise finished.
    resp = _request_with_retries(
        "POST", OVERPASS_URLS, data={"data": query}, headers=HEADERS, timeout=timeout + 30
    )
    elements = resp.json().get("elements", [])

    polygons: dict[str, list[Ring]] = {"parking": [], "industrial": [], "residential": [], "holes": []}
    seen_ids = set()
    for el in elements:
        el_type = el.get("type")
        if el_type not in ("way", "relation"):
            continue
        key = (el_type, el["id"])
        if key in seen_ids:
            continue
        seen_ids.add(key)

        tags = el.get("tags", {})
        if tags.get("amenity") == "parking":
            category = "parking"
        elif tags.get("building") in ("industrial", "warehouse") or tags.get("landuse") in ("industrial", "commercial"):
            category = "industrial"
        elif tags.get("landuse") == "residential":
            category = "residential"
        else:
            continue

        if el_type == "way":
            geometry = el.get("geometry")
            if not geometry or len(geometry) < 3:
                continue  # not a usable polygon (open way / missing nodes)
            polygons[category].append(_close_ring([(pt["lon"], pt["lat"]) for pt in geometry]))
        else:
            outer_rings, inner_rings = _relation_member_rings(el)
            polygons[category].extend(outer_rings)
            polygons["holes"].extend(inner_rings)

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
