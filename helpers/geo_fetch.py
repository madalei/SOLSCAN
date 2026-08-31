"""
Récupère les polygones géographiques qui servent de masques d'entraînement, depuis deux sources gratuites et sans clé API :

Fetch OSM (parking/industrial) and Cartofriches (friches) polygons for an AOI.

No API key needed for either source. See docs/roadmap_segmentation.md for the gotchas
this module works around (Overpass User-Agent, Cartofriches's former WFS being
Hauts-de-France-only). Overpass is a free, shared public instance and occasionally answers
with a 502/504 under load -- callers get retried automatically (mirror rotation) rather than
failing the whole AOI on a transient hiccup.
"""

import sqlite3
import struct
import time
from pathlib import Path

import requests

# Mirrors of the same public Overpass instance -- overpass-api.de alone times out often
# enough (504) that rotating through alternates on retry meaningfully improves success rate.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
]
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


# National Cartofriches extract (GeoPackage, ~67MB, ~36k sites), from data.gouv.fr's
# "Sites référencés dans Cartofriches" dataset. Replaces a per-AOI live WFS query against
# https://www.geo2france.fr: that GeoServer turned out to only mirror Hauts-de-France data
# (confirmed empirically -- querying it for departments 13/33/31/77 always returned 0
# features, even for genuinely friche-dense areas like Bouches-du-Rhone or Gironde) despite
# Cartofriches itself being a national Cerema dataset. This extract actually has that
# coverage (687 sites in dept 13, 311 in 33, 206 in 31, 150 in 77, checked directly) -- see
# docs/roadmap_segmentation.md.
CARTOFRICHES_GPKG_URL = "https://www.data.gouv.fr/api/1/datasets/r/a9084493-e742-4a2f-890b-0ebc803098df"
CARTOFRICHES_GPKG_PATH = Path(__file__).resolve().parent.parent / "data" / "cartofriches" / "cartofriches_national.gpkg"


def _ensure_cartofriches_gpkg(path: Path = CARTOFRICHES_GPKG_PATH, timeout: int = 300) -> Path:
    """Download the national Cartofriches GeoPackage once and cache it at `path` (under
    `data/`, gitignored). Downloaded straight to a `.part` file and renamed on success, so a
    crash mid-download can't leave a corrupt file that looks cached on the next run.
    """
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading national Cartofriches dataset (~67MB, one-time) to {path}...")
    tmp_path = path.with_suffix(".gpkg.part")
    with requests.get(CARTOFRICHES_GPKG_URL, headers=HEADERS, timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    tmp_path.rename(path)
    return path


def _parse_gpkg_polygon(blob: bytes) -> list[Ring]:
    """Parse a GeoPackage geometry blob (GPB header + WKB Polygon/MultiPolygon, 2D only)
    into a list of exterior rings. Interior rings (holes) are dropped -- same simplification
    already made for the OSM-sourced polygons in this module; no shapely/geopandas
    dependency in this project (see docs/roadmap_segmentation.md).
    """
    flags = blob[3]
    envelope_ind = (flags >> 1) & 0x07
    env_len = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[envelope_ind]
    wkb = blob[8 + env_len:]  # 4 bytes (magic+version+flags) + 4 bytes (srs_id) + envelope

    def read_rings(buf: bytes, offset: int, endian: str) -> tuple[list[Ring], int]:
        (num_rings,) = struct.unpack_from(endian + "I", buf, offset)
        offset += 4
        rings = []
        for _ in range(num_rings):
            (num_pts,) = struct.unpack_from(endian + "I", buf, offset)
            offset += 4
            coords = struct.unpack_from(endian + f"{num_pts * 2}d", buf, offset)
            offset += 16 * num_pts
            rings.append(list(zip(coords[0::2], coords[1::2])))
        return rings, offset

    endian = "<" if wkb[0] == 1 else ">"
    (geom_type,) = struct.unpack_from(endian + "I", wkb, 1)
    offset = 5
    exterior_rings: list[Ring] = []
    if geom_type == 3:  # Polygon
        rings, offset = read_rings(wkb, offset, endian)
        if rings:
            exterior_rings.append(rings[0])
    elif geom_type == 6:  # MultiPolygon -- each member is itself a full WKB Polygon (own byte-order+type header)
        (num_polys,) = struct.unpack_from(endian + "I", wkb, offset)
        offset += 4
        for _ in range(num_polys):
            sub_endian = "<" if wkb[offset] == 1 else ">"
            offset += 5
            rings, offset = read_rings(wkb, offset, sub_endian)
            if rings:
                exterior_rings.append(rings[0])
    return exterior_rings


def fetch_cartofriches_polygons(bbox: tuple[float, float, float, float]) -> list[Ring]:
    """Fetch friche (brownfield) polygons intersecting an AOI from the national Cartofriches
    extract, downloaded once and cached locally (see `_ensure_cartofriches_gpkg`).

    @param bbox: min_lon, min_lat, max_lon, max_lat (WGS84) -- queried directly against the
    GeoPackage's spatial (R-tree) index. No department/INSEE prefix needed: that was a
    workaround specific to the old WFS's unreliable BBOX param, not needed against a local
    file (see docs/roadmap_segmentation.md).
    """
    gpkg_path = _ensure_cartofriches_gpkg()
    min_lon, min_lat, max_lon, max_lat = bbox

    con = sqlite3.connect(gpkg_path)
    try:
        rows = con.execute(
            """
            SELECT f.geom FROM friches_surfaces f
            JOIN rtree_friches_surfaces_geom r ON f.fid = r.id
            WHERE r.minx <= ? AND r.maxx >= ? AND r.miny <= ? AND r.maxy >= ?
            """,
            (max_lon, min_lon, max_lat, min_lat),
        ).fetchall()
    finally:
        con.close()

    rings: list[Ring] = []
    for (blob,) in rows:
        rings.extend(_close_ring(ring) for ring in _parse_gpkg_polygon(blob))
    return rings
