"""Entry point: page config + sidebar navigation between the two model pages.

`streamlit run app/streamlit_app.py` -- see app/eurosat_page.py and app/unet_page.py for
the actual page content, app/map_widgets.py for the map components they share.
"""

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="SOLSCAN", layout="wide")
st.markdown(
    "**SOLSCAN** est un outil d'aide à la prospection photovoltaïque : il analyse des images "
    "satellite Sentinel-2 pour repérer automatiquement des zones industrielles susceptibles "
    "d'accueillir des projets solaires."
)

APP_DIR = Path(__file__).resolve().parent

eurosat_page = st.Page(APP_DIR / "eurosat_page.py", title="Classification EuroSAT", icon="🛰️", default=True)
unet_page = st.Page(APP_DIR / "unet_page.py", title="Segmentation U-Net", icon="🗺️")

pg = st.navigation([eurosat_page, unet_page])
pg.run()
