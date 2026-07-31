import base64
import io
import math
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import folium
import requests
import streamlit as st
from folium.plugins import Draw
from PIL import Image
from streamlit_folium import st_folium

sys.path.append(str(Path(__file__).resolve().parent.parent))
from helpers.palette import get_class_colors  # noqa: E402

API_BASE_URL = os.environ.get("SOLSCAN_API_URL", "http://localhost:8000")

st.set_page_config(page_title="SOLSCAN - Classification EuroSAT", layout="wide")
st.title("SOLSCAN - Classification de zone (EuroSAT / Sentinel-2)")
st.caption(
    "Dessine un rectangle sur la carte pour sélectionner une zone. "
    "L'API récupère la scène Sentinel-2 correspondante, découpe en tuiles 64x64 et classifie chaque tuile."
)


@st.cache_data(ttl=300)
def fetch_classes() -> list[str]:
    resp = requests.get(f"{API_BASE_URL}/classes", timeout=10)
    resp.raise_for_status()
    return resp.json()["classes"]


try:
    classes = fetch_classes()
except requests.RequestException:
    st.error(f"Impossible de joindre l'API sur {API_BASE_URL}. Démarre-la avec `uv run python -m uvicorn api.main:app --reload`.")
    st.stop()

st.sidebar.header("Paramètres")
today = date.today()
date_start = st.sidebar.date_input("Date de début", today - timedelta(days=180))
date_end = st.sidebar.date_input("Date de fin", today)
max_cloud_cover = st.sidebar.slider("Couverture nuageuse max (%)", 0, 100, 20)

m = folium.Map(location=[51.03, 2.37], zoom_start=13)
Draw(
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
).add_to(m)

map_data = st_folium(m, key="map", width=900, height=500)

MIN_ZOOM_FOR_SELECTION = 13  # matches the default zoom_start -> valid by default on load

current_zoom = map_data.get("zoom", MIN_ZOOM_FOR_SELECTION) if map_data else MIN_ZOOM_FOR_SELECTION
zoom_ok = current_zoom >= MIN_ZOOM_FOR_SELECTION
indicator_color = "#2ecc71" if zoom_ok else "#e74c3c"
indicator_text = (
    "Zoom OK — sélection possible"
    if zoom_ok
    else f"Zoom trop faible (niveau {current_zoom}, minimum {MIN_ZOOM_FOR_SELECTION}) — rapproche-toi sur la carte"
)
st.markdown(
    f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:8px'>"
    f"<div style='width:14px;height:14px;border-radius:50%;background-color:{indicator_color};"
    f"box-shadow:0 0 4px {indicator_color}'></div><span>{indicator_text}</span></div>",
    unsafe_allow_html=True,
)


def estimate_dimensions_km(bbox: tuple[float, float, float, float]) -> tuple[float, float, int]:
    """Rough width/height in km + tile count estimate (mirrors api/inference.py's guard-rail logic)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    earth_radius_m = 6_371_000
    lat_mid = math.radians((min_lat + max_lat) / 2)
    width_m = math.radians(max_lon - min_lon) * earth_radius_m * math.cos(lat_mid)
    height_m = math.radians(max_lat - min_lat) * earth_radius_m
    tile_m = 64 * 10  # EuroSAT tile: 64px at Sentinel-2's 10m/pixel
    n_tiles = math.ceil(width_m / tile_m) * math.ceil(height_m / tile_m)
    return width_m / 1000, height_m / 1000, n_tiles


bbox = None
if zoom_ok and map_data and map_data.get("last_active_drawing"):
    ring = map_data["last_active_drawing"]["geometry"]["coordinates"][0]
    lons = [pt[0] for pt in ring]
    lats = [pt[1] for pt in ring]
    drawn_bbox = (min(lons), min(lats), max(lons), max(lats))
    width_km, height_km, n_tiles = estimate_dimensions_km(drawn_bbox)
    if width_km * 1000 < 640 or height_km * 1000 < 640:
        st.warning(
            f"Rectangle trop petit ({width_km * 1000:.0f} x {height_km * 1000:.0f} m) — "
            "il faut au moins 640x640m (une tuile EuroSAT). Redessine un rectangle plus grand."
        )
    else:
        bbox = drawn_bbox
        st.caption(f"Zone sélectionnée : {width_km:.2f} x {height_km:.2f} km (~{n_tiles} tuiles de 640m).")
elif not zoom_ok and map_data and map_data.get("last_active_drawing"):
    st.caption("Rectangle dessiné avant un dézoom — invalide tant que le voyant n'est pas vert. Redessine une fois rapproché.")

classify_clicked = st.sidebar.button("Classifier", disabled=bbox is None)

if bbox is None:
    st.info("Dessine un rectangle sur la carte (outil rectangle en haut à gauche) pour activer la classification.")

if classify_clicked and bbox is not None:
    payload = {
        "bbox": list(bbox),
        "date_start": str(date_start),
        "date_end": str(date_end),
        "max_cloud_cover": max_cloud_cover,
    }
    with st.spinner("Classification en cours..."):
        try:
            resp = requests.post(f"{API_BASE_URL}/classify", json=payload, timeout=60)
        except requests.RequestException:
            st.error(f"Impossible de joindre l'API sur {API_BASE_URL}.")
            resp = None

    if resp is not None:
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", f"Erreur API ({resp.status_code})")
            except ValueError:
                detail = f"Erreur API ({resp.status_code})"
            st.error(detail)
        else:
            data = resp.json()
            original = Image.open(io.BytesIO(base64.b64decode(data["original_png_base64"])))
            overlay = Image.open(io.BytesIO(base64.b64decode(data["overlay_png_base64"])))

            col1, col2 = st.columns(2)
            col1.image(original, caption="Image Sentinel-2", use_container_width=True)
            col2.image(overlay, caption=f"Classification ({data['grid_cols']}x{data['grid_rows']} tuiles)", use_container_width=True)

            cloud_cover = data["cloud_cover_pct"]
            st.caption(f"Scène du {data['scene_datetime']} — couverture nuageuse {cloud_cover:.1f}%" if cloud_cover is not None else f"Scène du {data['scene_datetime']}")

            colors = get_class_colors(classes)
            legend_cols = st.columns(len(classes))
            for c, col in zip(classes, legend_cols):
                r, g, b = colors[c]
                col.markdown(
                    f"<div style='background-color:rgb({r},{g},{b});padding:4px;text-align:center;"
                    f"color:white;font-size:11px;border-radius:3px'>{c}</div>",
                    unsafe_allow_html=True,
                )

            st.subheader("Répartition par classe")
            table = [
                {"Classe": c, "Tuiles": data["tile_counts"][c], "%": data["tile_percentages"][c]}
                for c in classes
            ]
            st.dataframe(table, use_container_width=True, hide_index=True)
