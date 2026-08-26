"""EuroSAT classification page -- one label per 64px tile (10 classes)."""

import base64
import io
import os
import sys
from pathlib import Path

_app_dir = Path(__file__).resolve().parent
for _p in (_app_dir, _app_dir.parent):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

import requests
import streamlit as st
from PIL import Image

from helpers.image_utils import build_overlay  # noqa: E402
from helpers.palette import get_class_colors  # noqa: E402
from map_widgets import render_aoi_selector, render_result_map  # noqa: E402

API_BASE_URL = os.environ.get("SOLSCAN_API_URL", "http://localhost:8000")
TILE_SIZE_PX = 64

st.title("Classification EuroSAT")
st.caption(
    "Tracer un rectangle sur la carte pour sélectionner une zone puis cliquer sur 'Classifier'. "
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

# The AOI selector is a map widget that lets the user draw a rectangle on the map. The bbox is returned as (min_lon, min_lat, max_lon, max_lat).
bbox = render_aoi_selector(key="eurosat_map", tile_size_px=TILE_SIZE_PX)

classify_clicked = st.sidebar.button("Classifier", disabled=bbox is None)

if bbox is None:
    st.info("Dessine un rectangle sur la carte (outil rectangle en haut à gauche) pour activer la classification.")

if classify_clicked and bbox is not None:
    payload = {"bbox": list(bbox)}
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
            st.session_state["eurosat_result"] = resp.json()

# Rendering lives outside the "classify_clicked" branch so toggling the class filter below
# re-renders from the cached result instead of re-calling the API.
if "eurosat_result" in st.session_state:
    data = st.session_state["eurosat_result"]
    original = Image.open(io.BytesIO(base64.b64decode(data["original_png_base64"])))

    only_industrial = st.sidebar.toggle("Afficher uniquement la classe 'Industrial'", value=False)
    only_classes = {"Industrial"} if only_industrial else None

    tile_size = data["tile_size"]
    n_rows, n_cols = data["grid_rows"], data["grid_cols"]
    boxes = [
        (col * tile_size, row * tile_size, (col + 1) * tile_size, (row + 1) * tile_size)
        for row in range(n_rows)
        for col in range(n_cols)
    ]
    preds = [classes.index(label) for label in data["tile_labels"]]
    overlay = build_overlay(original, boxes, preds, classes, only_classes=only_classes, confidences=data["tile_confidences"])

    col1, col2 = st.columns(2)
    col1.image(original, caption="Image Sentinel-2", use_container_width=True)
    col2.image(overlay, caption=f"Classification ({n_cols}x{n_rows} tuiles)", use_container_width=True)

    cloud_cover = data["cloud_cover_pct"]
    st.caption(f"Scène du {data['scene_datetime']} — couverture nuageuse {cloud_cover:.1f}%" if cloud_cover is not None else f"Scène du {data['scene_datetime']}")

    legend_classes = [c for c in classes if not only_industrial or c == "Industrial"]
    colors = get_class_colors(classes)
    legend_cols = st.columns(len(legend_classes))
    for c, col in zip(legend_classes, legend_cols):
        r, g, b = colors[c]
        col.markdown(
            f"<div style='background-color:rgb({r},{g},{b});padding:4px;text-align:center;"
            f"color:white;font-size:11px;border-radius:3px'>{c}</div>",
            unsafe_allow_html=True,
        )

    st.subheader("Répartition par classe")
    table = [
        {"Classe": c, "Tuiles": data["tile_counts"][c], "%": data["tile_percentages"][c]}
        for c in legend_classes
    ]
    st.dataframe(table, use_container_width=True, hide_index=True)

    render_result_map(key="eurosat_result_map", bbox=data["bbox"], overlay_image=overlay)
