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



