# Heart Disease MLOps — End-to-End Pipeline

Production-style MLOps pipeline for predicting heart disease risk from patient clinical
data: data acquisition → EDA → feature engineering → model training with MLflow tracking →
packaged model → automated tests → CI/CD → containerized FastAPI service → Kubernetes
deployment → Prometheus/Grafana monitoring.

The model is intentionally simple (classical sklearn classifiers on a small tabular
dataset); the point of this project is the **pipeline** around it.

## Personalization / dataset choice

Rather than the widely-used 303-row Cleveland-only subset, this project combines **all
four original UCI collection sites** (Cleveland, Hungary, Switzerland, VA Long Beach —
920 rows, `src/data/download.py`). The combined set has real, site-correlated missingness
(`ca` 66% missing, `thal` 53%, `slope` 34%) that the clean subset doesn't have, which drove
real preprocessing decisions (see [`src/data/preprocess.py`](src/data/preprocess.py) and the
EDA notebook) rather than a trivial clean-data pass.

## Repository layout

```
data/               download script, raw (gitignored) + processed (committed) CSVs
notebooks/          01_eda.ipynb — executed EDA notebook
src/
  config.py          shared paths/constants
  data/              download + preprocessing pipeline
  models/            train.py, evaluate.py, predict.py
  api/               FastAPI app, schemas, structured logging
models/model.pkl     final packaged sklearn Pipeline (preprocessing + classifier)
mlruns/              local MLflow tracking store (gitignored, regenerate via train.py)
tests/               pytest unit tests (preprocessing, training, API)
docker/              Dockerfile + docker-compose.yml (api+prometheus+grafana)
monitoring/          Prometheus scrape config, Grafana provisioning + dashboard
k8s/                 raw manifests (k8s/manifests) + Helm chart (k8s/helm/heart-disease-api)
scripts/             build_and_run_docker.sh, deploy_minikube.sh
.github/workflows/   ci-cd.yml
reports/             screenshots, architecture diagram, model comparison, final report
```

## Setup (clean environment)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # includes requirements.txt + test/lint/notebook tools

python data/download_data.py          # -> data/raw/heart_disease_raw.csv (920 rows)
python -m src.data.preprocess         # -> data/processed/heart_disease_processed.csv
python -m src.models.train            # trains 3 models, logs to MLflow, saves models/model.pkl
```

Run the API locally:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,
  "thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6
}'
```

`slope`, `ca`, `thal` are optional in the request — the model was explicitly trained to
handle them missing (see the "Personalization" note above).

Inspect experiment tracking:

```bash
mlflow ui --backend-store-uri file:./mlruns   # http://127.0.0.1:5000
```

Run tests / lint (same commands CI runs):

```bash
ruff check src tests data
black --check src tests data
pytest -v --cov=src
```

## EDA & modelling summary

Full analysis: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb). Key points:

- Binary target is balanced (~55% disease / 45% no-disease) — no resampling needed.
- `thalach` (max heart rate) is negatively correlated with disease; `oldpeak`, `exang`,
  `sex`, `age` positively correlated — consistent with clinical literature.
- `ca`/`thal`/`slope` missingness is site-correlated (see EDA notebook §1), not random —
  handled with a constant sentinel category + one-hot encoding rather than mode imputation.

Three models were trained inside one `Pipeline` (preprocessing + classifier) each, tuned via
`GridSearchCV` over `RepeatedStratifiedKFold(n_splits=5, n_repeats=2)`, scored on ROC-AUC.
Held-out test set results (`reports/model_comparison.csv`):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV best ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.853 | 0.832 | 0.922 | 0.874 | 0.928 | 0.887 |
| Random Forest | 0.837 | 0.827 | 0.892 | 0.858 | 0.928 | 0.890 |
| **HistGradientBoosting (selected)** | **0.859** | **0.845** | **0.912** | **0.877** | **0.929** | 0.884 |

HistGradientBoosting was selected (best held-out ROC-AUC) and refit on the full dataset
with its tuned hyperparameters for the deployed artifact (`models/model.pkl`); the metrics
above come from the honest held-out split, not the full-data refit. Diagnostic plots
(confusion matrix, ROC, precision-recall curves per model) are in `reports/screenshots/` and
logged as MLflow artifacts.

## Experiment tracking (MLflow)

Every `GridSearchCV` run and the final full-data refit are logged as separate MLflow runs
under the `heart-disease-classification` experiment: params (best hyperparameters), CV and
test metrics, confusion/ROC/PR plots, and the fitted pipeline (`mlflow.sklearn.log_model`).
Run `mlflow ui --backend-store-uri file:./mlruns` to browse locally, or regenerate the whole
history with `python -m src.models.train`.

## CI/CD

[`​.github/workflows/ci-cd.yml`](.github/workflows/ci-cd.yml) — on every push/PR to `main`:

1. **lint-and-test**: ruff, black --check, pytest (coverage + JUnit artifacts).
2. **train**: downloads data, preprocesses, trains all 3 models on the runner (proves
   reproducibility from a clean checkout), uploads `model.pkl` + `mlruns/` + plots.
3. **docker-build-test**: builds the image using the freshly-trained model, runs the
   container, polls `/health`, smoke-tests `/predict`, and (on push to `main`) pushes to
   GHCR (`ghcr.io/<owner>/<repo>:latest` and `:<sha>`).

Any lint/test/training/build failure fails the pipeline — no `continue-on-error` anywhere.

## Docker

```bash
./scripts/build_and_run_docker.sh   # builds, runs, and curls /health + /predict
# or manually:
docker build -f docker/Dockerfile -t heart-disease-api:local .
docker run -p 8000:8000 heart-disease-api:local
```

Full monitoring stack (API + Prometheus + Grafana):

```bash
cd docker && docker compose up --build
# API:        http://localhost:8000
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin) — "Heart Disease API" dashboard is pre-provisioned
```

## Kubernetes (minikube)

```bash
./scripts/deploy_minikube.sh
```

This starts minikube (docker driver), enables the ingress addon, builds the image,
`minikube image load`s it, and deploys via Helm
(`k8s/helm/heart-disease-api`, or apply `k8s/manifests/*.yaml` directly with `kubectl`).
The script prints both an Ingress URL and a `kubectl port-forward` fallback.

## Monitoring & logging

- `/metrics` (Prometheus format): request counts by endpoint/status, request latency
  histogram, predictions by class.
- Structured JSON request logs to stdout (`src/api/logging_config.py`) — one line per
  request, container/K8s-log-aggregation friendly.
- Grafana dashboard: request rate, p95 latency, prediction-class mix, HTTP status codes
  (`monitoring/grafana/provisioning/dashboards/heart_disease_api_dashboard.json`).

## Report & video

- Full report: [`reports/MLOps_Heart_Disease_Report.docx`](reports/MLOps_Heart_Disease_Report.docx)
- Screenshots: [`reports/screenshots/`](reports/screenshots/)
- Video walkthrough script/checklist: [`VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md)
