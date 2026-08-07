"""U-Net landuse segmentation page -- pixel-level (fond / parking / industriel-commercial / friche).

Calls /v2/classify (api/segmentation_inference.py). The model isn't trained yet (see
docs/roadmap_segmentation.md), so the API returns 503 until a checkpoint exists.
"""

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

from helpers.mask_rasterize import CLASS_NAMES as LANDUSE_CLASSES  # noqa: E402
from helpers.mask_rasterize import CLASS_PREVIEW_COLORS as LANDUSE_COLORS  # noqa: E402
from map_widgets import render_aoi_selector, render_result_map  # noqa: E402

API_BASE_URL = os.environ.get("SOLSCAN_API_URL", "http://localhost:8000")
TILE_SIZE_PX = 256

st.title("Segmentation U-Net (landuse)")
st.caption(
    "Tracer un rectangle sur la carte puis cliquer sur 'Segmenter'. Segmentation pixel par "
    "pixel (fond / parking / industriel-commercial / friche) -- modèle pas encore entraîné, "
    "l'API répond une erreur 503 tant que ce n'est pas fait."
)

st.sidebar.header("Paramètres")

bbox = render_aoi_selector(key="unet_map", tile_size_px=TILE_SIZE_PX)

classify_clicked = st.sidebar.button("Segmenter", disabled=bbox is None)

if bbox is None:
    st.info("Dessine un rectangle sur la carte (outil rectangle en haut à gauche) pour activer la segmentation.")

if classify_clicked and bbox is not None:
    payload = {"bbox": list(bbox)}
    with st.spinner("Segmentation en cours..."):
        try:
            resp = requests.post(f"{API_BASE_URL}/v2/classify", json=payload, timeout=60)
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
            st.session_state["unet_result"] = resp.json()

if "unet_result" in st.session_state:
    data = st.session_state["unet_result"]
    original = Image.open(io.BytesIO(base64.b64decode(data["original_png_base64"])))
    overlay = Image.open(io.BytesIO(base64.b64decode(data["overlay_png_base64"])))

    col1, col2 = st.columns(2)
    col1.image(original, caption="Image Sentinel-2", use_container_width=True)
    col2.image(overlay, caption=f"Segmentation ({data['grid_cols']}x{data['grid_rows']} tuiles)", use_container_width=True)

    cloud_cover = data["cloud_cover_pct"]
    st.caption(f"Scène du {data['scene_datetime']} — couverture nuageuse {cloud_cover:.1f}%" if cloud_cover is not None else f"Scène du {data['scene_datetime']}")

    legend_cols = st.columns(len(LANDUSE_CLASSES))
    for class_id, (name, col) in enumerate(zip(LANDUSE_CLASSES, legend_cols)):
        r, g, b = LANDUSE_COLORS[class_id]
        col.markdown(
            f"<div style='background-color:rgb({r},{g},{b});padding:4px;text-align:center;"
            f"color:white;font-size:11px;border-radius:3px'>{name}</div>",
            unsafe_allow_html=True,
        )

    st.subheader("Répartition par classe (% de pixels)")
    table = [
        {"Classe": name, "Pixels": data["class_pixel_counts"][name], "%": data["class_pixel_percentages"][name]}
        for name in LANDUSE_CLASSES
    ]
    st.dataframe(table, use_container_width=True, hide_index=True)

    render_result_map(key="unet_result_map", bbox=data["bbox"], overlay_image=overlay)
