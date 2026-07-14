# chrono24_price_prediction

Prediction of price of Top Luxury Watches on chrono24 Dataset based on their information

https://www.kaggle.com/datasets/philmorekoung11/luxury-watch-listings

The dataset contains over 280,000 listings of luxury watches including features such as list price, model, reference number, and case material.

The rule of branch and commit naming:

```
git checkout -b feature/<yourname>-<short-description>
```
```
git commit -m "[Feature/fix/..] Name of feature/fix/.."
```

## Project structure

```
chrono24_price_prediction/
├── .github/workflows/ci.yml     # CI: ruff (lint + format) + pytest on every push
├── notebooks/                   # where the exploration happens (EDA, experiments)
├── src/chrono24/                # the pipeline as an importable package
│   ├── data.py                  #   load_watches() — one loader for Kaggle / Colab / local
│   ├── cleaning.py              #   parsing, deduplication, key-column filters
│   ├── outliers.py              #   range rules, price clipping, IQR, rare categories
│   ├── features.py              #   feature engineering → X, y
│   ├── training.py              #   LightGBM training, metrics, save/load
│   ├── plots.py                 #   evaluation figures
│   └── __main__.py              #   python -m chrono24 → full pipeline
├── models/                      # trained model + metrics (committed)
├── reports/figures/             # evaluation plots (committed)
├── data/                        # git-ignored datasets (see data/README.md)
├── tests/                       # pytest unit + smoke tests
├── Dockerfile                   # reproducible environment image
├── Makefile                     # one command per task
├── pyproject.toml               # dependencies + tooling (uv)
└── uv.lock                      # pinned, reproducible environment
```

## Using the shared code in a notebook (Kaggle / Colab)

We work in notebooks and keep reusable code in the `chrono24` package so it isn't
copy-pasted around. Install it at the top of a notebook and import what you need:

```python
!pip install -q git+https://github.com/Glazochek/chrono24_price_prediction.git@develop

from chrono24 import load_watches
df = load_watches()      # finds the data on Kaggle, Colab, or local automatically
```

`load_watches()` looks for the dataset in order: the mounted dataset on Kaggle, a local
copy in `data/raw/`, then a Kaggle download (Colab / local, needs credentials).

## Local development

Requires [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
uv sync                        # create the .venv and install from uv.lock
uv run pre-commit install      # enable the ruff commit hooks
cp .env.example .env           # add your Kaggle credentials for downloads
make data                      # download the dataset and print its shape
```

Kaggle credentials (`KAGGLE_USERNAME` / `KAGGLE_KEY`) go in `.env` — see `.env.example`.
`.env` and everything under `data/` are git-ignored.

## Commands (`make help`)

| Command | What it does |
|---|---|
| `make setup`  | Install the environment from the lockfile |
| `make lint`   | Ruff check + format check (same as CI) |
| `make format` | Auto-fix and format the code |
| `make test`   | Run the test suite |
| `make data`   | Download the dataset and print its shape |
| `make train`  | Run the full pipeline and save model + metrics + plots |
| `make docker` | Build the container image |
| `make clean`  | Remove caches |

## Training pipeline

`make train` (or `uv run python -m chrono24`) runs the whole pipeline from the notebook,
end to end: load → clean → remove outliers → engineer features → train LightGBM →
evaluate. It writes:

- `models/model.txt` — the trained model (LightGBM booster)
- `models/metrics.json` — RMSE / MAE / R² in log space and dollars
- `reports/figures/*.png` — loss curve, residuals, predicted-vs-actual, feature importance

To load the committed model and predict:

```python
from chrono24 import load_model, predict_price

model = load_model("models/model.txt")
prices = predict_price(model, X)   # X built with chrono24.build_features
```

## Docker

```bash
make docker
docker run --rm -v "$(pwd)/data:/app/data" chrono24
```

A reproducible image with Python + all dependencies. Mount `data/` (or pass Kaggle
credentials) so it can load the dataset.

## Continuous integration

Every push to `develop` and every PR into `main` / `develop` runs
[`.github/workflows/ci.yml`](.github/workflows/ci.yml): install from the lockfile → Ruff
lint & format check → pytest.
