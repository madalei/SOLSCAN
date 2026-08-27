# SOLSCAN — Application de télédetection 

**SOLSCAN** est un outil d'aide à la prospection photovoltaïque : il analyse des images satellite Sentinel-2 pour repérer automatiquement des zones déjà artificialisées (friches, sites industriels, grands parkings) susceptibles d'accueillir des projets solaires.
Le moteur de classification (ou segmentation selon la version) utilise un modele de Deep Learning.

## Quick start

**Require Docker installed**

Clone the Repo <br>
`git clone https://github.com/madalei/SOLSCAN.git`

Open folder <br>
`cd SOLSCAN`

Checkout the branch ou want to explore (unet recommanded) <br>
`git checkout main` or `git checkout unet`

Start with Docker <br>
`docker compose up -d --build`

Or start manually with `uv`
```
uv run python -m uvicorn api.main:app --reload --port 8000
uv run streamlit run app/streamlit_app.py
```
Then open a browser and find your running app at `http://localhost:8501/`


## Stack actuelle

1. **Frontend Streamlit** (`app/streamlit_app.py`) — carte satellite interactive, on dessine un rectangle (zone d'intérêt), l'app appelle l'API.
2. **API FastAPI** (`api/main.py`) — récupère la scène Sentinel-2 correspondante (via Microsoft Planetary Computer), découpe en tuiles, classifie, renvoie une image overlay colorée + statistiques par classe.
3. **Modèle** — un ResNet18 fine-tuné sur EuroSAT (10 classes de type d'occupation des sols, dont "Industrial") sert de preuve de pipeline : classification tuile entière (64px), pas de vraies frontières pixel.

## Branche `main`

Développement initial. Utilise un modele de Deep learning basé sur ResNet18 et fine-tuné sur les classes Eurosat. Cette version aboutit a une classification des zones par carrés de 640m de coté.

## Branche `unet` (en cours)

Segmentation multi-classe (fond/parking/industriel/friche) via U-Net, entraînée sur des masques dérivés d'OpenStreetMap + Cartofriches, pour obtenir des frontières pixel précises et une estimation de surface fiable — l'objectif final du projet

## Contexte

Projet réalisé dans le cadre d'une certification RNCP (data science / deep learning) — d'où les notebooks de comparaison de modèles/losses/stratégies d'entraînement, qui documentent la démarche autant qu'ils produisent un résultat.


### Arborescence du projet

```
SOLSCAN/
├── api/                          # Backend FastAPI
│   ├── main.py                   # Endpoints /classify (EuroSAT) et /v2/classify (U-Net)
│   ├── inference.py               # Fetch Sentinel-2, classification par tuile (EuroSAT), overlay
│   ├── segmentation_inference.py  # Segmentation par tuile (U-Net), overlay du masque
│   └── schemas.py                 # Modèles Pydantic des requêtes/réponses de l'API
├── app/                          # Frontend Streamlit
│   ├── streamlit_app.py          # Point d'entrée, sélecteur de modèle (EuroSAT / U-Net)
│   ├── eurosat_page.py            # Vue carte + résultats pour le classifieur EuroSAT
│   ├── unet_page.py                # Vue carte + résultats pour le U-Net
│   └── map_widgets.py              # Utilitaires carte communs aux deux vues
├── models/                       # Constructeurs de modèles (architecture + poids pré-entraînés)
│   ├── resnet18_classifier_builder.py
│   ├── efficientnet_b0_classifier_builder.py
│   └── unet_builder.py            # U-Net (segmentation_models_pytorch), encodeur ResNet34
├── training/                     # Boucles d'entraînement et d'évaluation
│   ├── engine.py                  # Boucle générique (classification EuroSAT)
│   ├── seg_engine.py               # SegEngine : loss Dice + CE pondérée, IoU par classe (U-Net)
│   ├── evaluate.py                 # Rapport de classification, matrice de confusion
│   └── pipeline_configs.py         # Config des expériences (RESNET18_BASELINE, UNET_CONFIG)
├── helpers/                      # Fonctions utilitaires partagées
│   ├── dataloaders.py             # Chargement EuroSAT + split train/val/test générique
│   ├── segmentation_dataset.py     # Dataset PyTorch pour les tuiles landuse (image + masque)
│   ├── geo_fetch.py                 # Requêtes OpenStreetMap (Overpass) + Cartofriches (WFS)
│   ├── mask_rasterize.py            # Polygones géo -> masque raster multi-classe
│   ├── landuse_aois.py               # Zones géographiques utilisées pour bâtir le dataset U-Net
│   ├── image_utils.py                 # Recadrage/découpe d'images en tuiles
│   └── palette.py                      # Couleurs par classe pour les overlays
├── notebooks/                    # Exploration, entraînement, benchmark (documentent la démarche)
│   ├── fetch_eurosat.ipynb
│   ├── train_eurosat_classifier.ipynb  # Comparaison de 4 variantes ResNet18 / EfficientNet-B0
│   ├── tile_grid_classification.ipynb
│   ├── fetch_landuse_dataset.ipynb     # Génère le dataset d'entraînement du U-Net
│   └── train_unet_landuse.ipynb        # Entraîne le U-Net, suit l'IoU par classe
├── data/                          # Données (générées ou téléchargées, volumineuses)
│   ├── eurosat/                    # Dataset EuroSAT (27 000 tuiles, 10 classes)
│   ├── landuse/                     # Tuiles Sentinel-2 + masques générés (images/masks/masks_preview)
│   └── samples/                      # Image d'exemple pour tests manuels
├── checkpoints/                   # Poids des modèles entraînés (.pth), committés pour le déploiement
├── docs/
│   ├── glossaire.md                 # Définitions des concepts ML/géo utilisés dans le projet
│   ├── roadmap_segmentation.md       # Historique/diagnostic du premier essai U-Net
│   └── soutenance/                    # Livret d'apprentissage RNCP
├── docker-compose.yml              # Orchestration des 2 services (api, app)
├── main.py                         # Stub non utilisé (généré par `uv init`)
└── pyproject.toml / uv.lock        # Dépendances du projet, gérées avec uv
```

## Developpement mode 

### Installation -- How I installed, for memo

The project use uv as dependency manager and project helper

Create a new project (generates pyproject.toml, .venv, etc.) <br>
`uv init`

Add dependencies (writes to pyproject.toml + uv.lock, installs into .venv) <br>
`uv add pandas numpy scikit-learn`


Add a dev-only dependency <br>
`uv add --dev pytest`

Run something inside the project's venv without manually activating it <br>
`uv run python main.py`
`uv run pytest`

Sync the venv to match the lockfile exactly (e.g. after cloning the repo) <br>
`uv sync`


### Run a Notebook

`uv run jupyter notebook notebooks/fetch_eurosat.ipynb`

### Run API and front app

`uv run python -m uvicorn api.main:app --reload --port 8000`
`uv run streamlit run app/streamlit_app.py`

### Run in Debug mode in VS code

Launch configs are edited in `.vscode/launch.json`

Open Run & Debug (⇧⌘D), you see configs

FastAPI: api.main — lance uvicorn api.main:app --reload in debug mode

Streamlit: app/streamlit_app.py — lance l'app Streamlit in debug mode

Python: Current File — debug n'importe quel script Python ouvert 

## Deploy on a server with Docker

Two services (`api`: FastAPI + model, `app`: Streamlit), orchestrated via `docker-compose.yml`. CPU-only torch (see `[tool.uv.sources]` in `pyproject.toml`) -- no GPU needed or used.

#### Prerequisites on the server
- Docker + the Compose plugin installed (`docker compose version` to check).

#### Checkout code, build and run
```
git clone https://github.com/madalei/SOLSCAN.git
cd SOLSCAN
git checkout <branch>
docker compose up -d --build
```
The model checkpoint (`checkpoints/resnet18_eurosat.pth`) must be committed to git 

### Ping and check you get no error
```
curl http://<server>:8000/classes
```
Then open `http://<server>:8501` in a browser for the Streamlit app. Make sure ports 8000 and 8501 are open in the server's firewall/security group (most small VPS block everything but SSH by default).

#### Useful commands
- Logs: `docker compose logs -f`
- Stop: `docker compose down`
- Rebuild after a code change: `docker compose up -d --build`

#### Known limitation
Both images currently install the full project dependency set rather than being split per-service, so each is ~4.3GB (fine on most small VPS with 20GB+ disk, but not lean). Ask if this needs slimming down.
