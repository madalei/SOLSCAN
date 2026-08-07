"""Map widgets shared by the EuroSAT and U-Net pages: the AOI-drawing map and the
georeferenced result map. Both pages need the exact same "draw a rectangle, validate
zoom/size, plot the overlay back on a real map" flow -- only the tile size (and therefore
the minimum AOI size) differs between them.
"""

import math

import folium
import numpy as np
import streamlit as st
from branca.element import MacroElement, Template
from folium.plugins import Draw, MousePosition
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium

MIN_ZOOM_FOR_SELECTION = 13  # matches the default zoom_start below -> valid by default on load
SENTINEL2_GSD_M = 10  # meters/pixel


class _ClearPreviousDrawing(MacroElement):
    """Wipe any earlier drawn shape as soon as a new one is completed.

    streamlit-folium extracts each map child's own `script` macro (it doesn't render
    Figure.script), so this has to be a real MacroElement child of the map -- a plain
    folium.Element added to the root Figure is silently dropped by its rendering pipeline.
    """

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        {{ this._parent.get_name() }}.on('draw:created', function (e) {
            var newLayer = e.layer;
            drawnItems_{{ this.draw_name }}.clearLayers();
            drawnItems_{{ this.draw_name }}.addLayer(newLayer);
        });
        {% endmacro %}
        """
    )

    def __init__(self, draw_name: str):
        super().__init__()
        self._name = "ClearPreviousDrawing"
        self.draw_name = draw_name


def _estimate_dimensions_km(bbox: tuple[float, float, float, float], tile_m: float) -> tuple[float, float, int]:
    """Rough width/height in km + tile count estimate (mirrors the API's guard-rail logic)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    earth_radius_m = 6_371_000
    lat_mid = math.radians((min_lat + max_lat) / 2)
    width_m = math.radians(max_lon - min_lon) * earth_radius_m * math.cos(lat_mid)
    height_m = math.radians(max_lat - min_lat) * earth_radius_m
    n_tiles = math.ceil(width_m / tile_m) * math.ceil(height_m / tile_m)
    return width_m / 1000, height_m / 1000, n_tiles


def render_aoi_selector(
    key: str,
    tile_size_px: int,
    location: tuple[float, float] = (51.03, 2.37),
    zoom_start: int = 13,
) -> tuple[float, float, float, float] | None:
    """Draw-a-rectangle map + zoom guard + minimum-size check.

    @param tile_size_px: model tile size in pixels (64 for EuroSAT, 256 for U-Net) --
    drives the minimum AOI size warning and the tile-count estimate.
    @return the validated (min_lon, min_lat, max_lon, max_lat) bbox, or None if nothing
    valid is selected yet.
    """
    tile_m = tile_size_px * SENTINEL2_GSD_M

    m = folium.Map(location=list(location), zoom_start=zoom_start, tiles="Esri.WorldImagery")
    draw = Draw(
        export=False,
        draw_options={
            "rectangle": True,
            "polygon": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "polyline": False,
        },
        edit_options={"edit": False},
    )
    draw.add_to(m)
    # Only one rectangle should be visible at a time: as soon as a new one is completed,
    # clear whatever was drawn before it instead of letting old selections pile up.
    _ClearPreviousDrawing(draw.get_name()).add_to(m)
    MousePosition(position="bottomleft", separator=" | ", prefix="Coordonnées :", num_digits=5).add_to(m)

    map_data = st_folium(m, key=key, width=900, height=500)

    current_zoom = map_data.get("zoom", zoom_start) if map_data else zoom_start
    zoom_ok = current_zoom >= MIN_ZOOM_FOR_SELECTION
    indicator_color = "#2ecc71" if zoom_ok else "#e74c3c"
    indicator_text = (
        "Zoom OK — sélection possible"
        if zoom_ok
        else f"Zoom trop faible (niveau {current_zoom}, minimum {MIN_ZOOM_FOR_SELECTION}) — zoomer sur la carte"
    )
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:8px'>"
        f"<div style='width:14px;height:14px;border-radius:50%;background-color:{indicator_color};"
        f"box-shadow:0 0 4px {indicator_color}'></div><span>{indicator_text}</span></div>",
        unsafe_allow_html=True,
    )

    bbox = None
    if zoom_ok and map_data and map_data.get("last_active_drawing"):
        ring = map_data["last_active_drawing"]["geometry"]["coordinates"][0]
        lons = [pt[0] for pt in ring]
        lats = [pt[1] for pt in ring]
        drawn_bbox = (min(lons), min(lats), max(lons), max(lats))
        width_km, height_km, n_tiles = _estimate_dimensions_km(drawn_bbox, tile_m)
        if width_km * 1000 < tile_m or height_km * 1000 < tile_m:
            st.warning(
                f"Rectangle trop petit ({width_km * 1000:.0f} x {height_km * 1000:.0f} m) — "
                f"il faut au moins {tile_m:.0f}x{tile_m:.0f}m (une tuile). Redessine un rectangle plus grand."
            )
        else:
            bbox = drawn_bbox
            st.caption(f"Zone sélectionnée : {width_km:.2f} x {height_km:.2f} km (~{n_tiles} tuiles de {tile_m:.0f}m).")
    elif not zoom_ok and map_data and map_data.get("last_active_drawing"):
        st.caption("Rectangle dessiné avant un dézoom — invalide tant que le voyant n'est pas vert. Redessine une fois rapproché.")

    return bbox


def render_result_map(key: str, bbox: tuple[float, float, float, float], overlay_image) -> None:
    """OSM/Satellite map with the classification/segmentation overlay geo-referenced on bbox."""
    st.subheader("Zones classifiées sur la carte")
    st.caption(
        "L'overlay est géoréférencé sur ses coordonnées réelles. L'image est recadrée à un "
        "nombre entier de tuiles côté API, donc ses bords peuvent différer de quelques "
        "dizaines de mètres du rectangle dessiné plus haut."
    )
    min_lon, min_lat, max_lon, max_lat = bbox
    result_map = folium.Map(location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2], zoom_start=15)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(result_map)
    folium.TileLayer("Esri.WorldImagery", name="Satellite (Esri)", attr="Esri").add_to(result_map)
    ImageOverlay(
        image=np.array(overlay_image),
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        opacity=0.8,
        name="Classification",
    ).add_to(result_map)
    folium.LayerControl(collapsed=False).add_to(result_map)
    MousePosition(position="bottomleft", separator=" | ", prefix="Coordonnées :", num_digits=5).add_to(result_map)
    st_folium(result_map, key=key, width=900, height=500, returned_objects=[])
