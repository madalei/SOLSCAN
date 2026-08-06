### Installation -- How I installed, for memo

#### Create a new project (generates pyproject.toml, .venv, etc.)
`uv init`

#### Add dependencies (writes to pyproject.toml + uv.lock, installs into .venv)
`uv add pandas numpy scikit-learn`


#### Add a dev-only dependency
`uv add --dev pytest`

#### Run something inside the project's venv without manually activating it
`uv run python main.py`
`uv run pytest`

#### Sync the venv to match the lockfile exactly (e.g. after cloning the repo)
`uv sync`


#### Run a Notebook

`uv run jupyter notebook notebooks/fetch_eurosat.ipynb`

#### Run API and front app

`uv run python -m uvicorn api.main:app --reload --port 8000`
`uv run streamlit run app/streamlit_app.py`

### Deploy on a server (Docker)

Two services (`api`: FastAPI + model, `app`: Streamlit), orchestrated via `docker-compose.yml`. CPU-only torch (see `[tool.uv.sources]` in `pyproject.toml`) -- no GPU needed or used.

#### Prerequisites on the server
- Docker + the Compose plugin installed (`docker compose version` to check).

#### Get the code and build
```
git clone <repo-url>
cd SOLSCAN
docker compose up -d --build
```
The model checkpoint (`checkpoints/resnet18_eurosat.pth`) is committed to git (deliberately overrides the usual `*.pth` gitignore rule) specifically so a plain clone is enough -- nothing else to transfer to the server.

#### Verify
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
