# Loan Prediction Approval — MLOps Project

> ENSAE Paris — *Mise en production* course 

End-to-end MLOps pipeline for predicting loan approval: data processing, hyperparameter tuning across three model families, MLflow experiment tracking, FastAPI deployment, Kubernetes orchestration on SSPCloud, GitOps automation via ArgoCD, Prometheus/Grafana monitoring, SHAP explanations, and drift-triggered automatic retraining.

---

## Live URLs (SSPCloud)

Replace `<username>` with your SSPCloud username everywhere below.

| Service | URL pattern |
|---------|-------------|
| **Web UI** | `https://loan-api-<username>.user.lab.sspcloud.fr` |
| **Swagger UI** | `https://loan-api-<username>.user.lab.sspcloud.fr/docs` |
| **Prometheus metrics** | `https://loan-api-<username>.user.lab.sspcloud.fr/metrics` |
| **Grafana dashboard** | `https://grafana-loan-<username>.user.lab.sspcloud.fr` (admin / admin) |
| **MLflow UI** | `https://mlflow-<username>.user.lab.sspcloud.fr` |

Your username is the prefix of your SSPCloud namespace (e.g. namespace `user-johndoe` → username `johndoe`).

---

## Reproduce from scratch

This section is for anyone who wants to clone the repo and get the exact same results.

### Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | **3.13** | https://www.python.org/downloads/ |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker + Docker Compose | any recent | https://docs.docker.com/get-docker/ |
| git | any | — |

Optional (Kubernetes deployment only):
- `kubectl` configured against an SSPCloud cluster
- Access to a MinIO S3 bucket on `minio.lab.sspcloud.fr`

---

### Step 1 — Clone and install

```bash
git clone https://github.com/kellybourbon2/Loan-prediction-approval.git
cd Loan-prediction-approval
uv sync                        # installs exact locked dependencies (uv.lock)
uv run pre-commit install      # enables ruff lint+format on every commit
```

`uv sync` reads `uv.lock` — every dependency is pinned, so you get the identical environment.

---

### Step 2 — Dataset

The model trains on the **Kaggle Playground Series S4E10 — Loan Approval Prediction** dataset.

1. Download `train.csv` from https://www.kaggle.com/competitions/playground-series-s4e10/data
2. Upload it to your S3 bucket at the root: `s3://<your-bucket>/train.csv`

The data loader reads it directly from S3 at training time — no local copy needed.

---

### Step 3 — Environment variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
AWS_ACCESS_KEY_ID=<your_key>
AWS_SECRET_ACCESS_KEY=<your_secret>
AWS_SESSION_TOKEN=<your_token>         # leave empty if not using SSPCloud temp tokens
AWS_S3_ENDPOINT=minio.lab.sspcloud.fr
AWS_BUCKET_NAME=<your_bucket>          # the bucket where train.csv is stored
```

These variables are loaded automatically by `data_load.py` via `python-dotenv`.

---

### Step 4 — Train the model

```bash
uv run python src/main.py
```

What happens:

| Step | Detail |
|------|--------|
| Load | `train.csv` downloaded from S3 |
| Preprocess | `DataPreprocessor`: drop unused columns, bin age, binary-encode credit default, StandardScaler + OneHotEncoder |
| Split | **3-way**: 80% training / 10% calibration / 10% evaluation — no leakage between the three |
| Tune | Hyperopt TPE search (`MAX_EVALS=10`) over **XGBoost**, **CatBoost**, **RandomForest** simultaneously — best model wins |
| Train | Best model retrained on full training set |
| Calibrate | `CalibratedClassifierCV(method='isotonic', cv='prefit')` fitted on the calibration split — well-calibrated probabilities |
| Evaluate | Accuracy, F1, Recall, Precision + confusion matrix on the eval split (never seen before) |
| Log | All metrics, params, confusion matrix PNG artifact → MLflow experiment `Loan Prediction Approval Experiments` |
| Register | Model registered in MLflow Registry as `@challenger` |
| Promote | Promoted to `@champion` **only if** F1 ≥ 0.5 **and** F1 > current champion (regression guard) |

Expected results on the eval split (may vary slightly due to Hyperopt stochasticity):

| Metric | Typical value |
|--------|--------------|
| F1 | ~0.88 |
| Accuracy | ~0.93 |
| Recall | ~0.87 |
| Precision | ~0.89 |

To inspect runs after training:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
# → open http://127.0.0.1:5000
```

---

### Step 5 — Run the API locally

The API loads the `@champion` model from MLflow at startup.

```bash
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

Test it:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "person_age": 30,
    "person_income": 60000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5.0,
    "loan_intent": "PERSONAL",
    "loan_amnt": 10000,
    "loan_percent_income": 0.17,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 4
  }'
# → {"loan_status": 1, "approved": true, "probability": 0.8742}
```

---

### Step 6 — Run the full local stack (API + Prometheus + Grafana)

```bash
docker compose up
```

| Service | URL | Credentials |
|---------|-----|-------------|
| API | http://localhost:8000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

The Grafana datasource and dashboards are provisioned automatically on first start.

---

### Step 7 — Run the tests

```bash
# Unit tests — mocked model, no network needed (28 tests)
uv run pytest unit_tests/ --ignore=unit_tests/test_integration.py -v

# Integration tests — requires a running API
INTEGRATION_API_URL=http://localhost:8000 \
  uv run pytest unit_tests/test_integration.py -v
```

| Suite | Tests | What is covered |
|-------|-------|-----------------|
| `test_preprocessing.py` | 6 | `DataPreprocessor`: clean, feature engineering, split, encoding |
| `test_api.py` | 14 | `/predict`, `/predict/batch`, `/explain` — mocked model |
| `test_integration.py` | 8 | Real HTTP calls — health, predict, batch, explain, metrics, no traceback leak |

---

## Repository structure

```
├── src/
│   ├── api/
│   │   ├── app.py            # FastAPI app (predict, batch, explain, health, metrics)
│   │   ├── schemas.py        # Pydantic input/output schemas
│   │   ├── metrics.py        # Prometheus metrics definitions
│   │   └── logger.py         # Structured prediction logger → S3 sync
│   ├── model/
│   │   ├── train.py          # Model training wrapper
│   │   ├── tune.py           # Hyperopt objective + model builder (with early stopping)
│   │   ├── evaluate.py       # Metrics + confusion matrix → MLflow
│   │   ├── registry.py       # MLflow registry: register, promote, load champion
│   │   └── search_space.py   # Hyperopt search space (XGBoost, CatBoost, RF)
│   ├── data_processing/
│   │   ├── preprocessing.py  # DataPreprocessor (clean → engineer → scale → encode)
│   │   └── data_load.py      # S3 data loading via s3fs
│   ├── main.py               # Full training entrypoint
│   └── drift_analysis.py     # KS test + PSI drift detection
├── .github/workflows/
│   ├── ci.yml                # Ruff + unit tests + integration tests
│   ├── cd.yml                # Build Docker → push → update k8s manifest → healthcheck → rollback
│   ├── retrain.yml           # Manual/scheduled retraining (every Monday 2am UTC)
│   └── drift_check.yml       # Daily drift check → triggers retrain if drift detected
├── k8s/                      # Kubernetes manifests (ArgoCD GitOps)
├── monitoring/
│   └── grafana/
│       ├── dashboards/       # Dashboard JSON (auto-provisioned)
│       └── provisioning/     # Datasources, dashboards, alerting rules
├── unit_tests/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml            # Python project + ruff config
├── uv.lock                   # Pinned dependency lockfile
└── config.py                 # All training constants (CV_FOLDS, MAX_EVALS, thresholds…)
```

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` or `/ui/` | Web UI — loan assessment form |
| `GET` | `/health` | Returns 200 if model loaded, 503 otherwise |
| `POST` | `/predict` | Single loan prediction |
| `POST` | `/predict/batch` | Batch prediction (max 500 per request) |
| `POST` | `/explain` | SHAP feature contributions for one application |
| `GET` | `/metrics` | Prometheus metrics endpoint |
| `GET` | `/docs` | Swagger UI |

### SHAP explanations

SHAP values are computed without the external `shap` library (incompatible with Python 3.13):
- **XGBoost**: `get_booster().predict(dmat, pred_contribs=True)`
- **CatBoost**: `get_feature_importance(type="ShapValues", data=pool)`
- **RandomForest**: global feature importances weighted by prediction deviation

Positive SHAP values push toward approval, negative toward rejection.

---

## CI/CD

| Workflow | Trigger | What it does |
|----------|---------|-------------|
| `ci.yml` | Every push | Ruff lint + format check → unit tests → integration tests against `API_URL` |
| `cd.yml` | Push touching `src/`, `Dockerfile`, `pyproject.toml`, `uv.lock` | Build Docker image → push to Docker Hub → update `k8s/deployment.yaml` image tag → wait 60s → GET `/health` → auto-rollback if 503 |
| `retrain.yml` | Manual or every Monday 2am UTC | Full retraining + MLflow registry update |
| `drift_check.yml` | Daily 8am UTC | Download `predictions.jsonl` from S3 → KS + PSI analysis → trigger `retrain.yml` if drift detected |

### Required GitHub Actions configuration

Go to **Settings → Secrets and variables → Actions** and add:

| Name | Type | Value |
|------|------|-------|
| `DOCKERHUB_TOKEN` | Secret | Docker Hub access token |
| `AWS_ACCESS_KEY_ID` | Secret | S3 credentials (for retrain + drift check) |
| `AWS_SECRET_ACCESS_KEY` | Secret | — |
| `AWS_SESSION_TOKEN` | Secret | — |
| `AWS_S3_ENDPOINT` | Secret | e.g. `minio.lab.sspcloud.fr` |
| `AWS_BUCKET_NAME` | Secret | — |
| `GH_PAT` | Secret | (optional) GitHub PAT with `repo` scope|
| `DOCKERHUB_USERNAME` | Variable | Docker Hub username |
| `API_URL` | Variable | Deployed API base URL — enables integration tests and post-deploy healthcheck |

---

## Kubernetes deployment (SSPCloud)

### One-time secrets setup

Before applying, update `k8s/deployment.yaml` and `k8s/ingress.yaml`: replace every occurrence of `user-oualy` with your own namespace (`user-<username>`).

```bash
# AWS / MinIO credentials used by the API at runtime
kubectl create secret generic loan-api-secret \
  --from-literal=AWS_ACCESS_KEY_ID=<key> \
  --from-literal=AWS_SECRET_ACCESS_KEY=<secret> \
  --from-literal=AWS_SESSION_TOKEN=<token> \
  --from-literal=AWS_S3_ENDPOINT=https://minio.lab.sspcloud.fr \
  --from-literal=AWS_BUCKET_NAME=<bucket> \
  -n user-<username>
```

### Deploy

```bash
kubectl apply -f k8s/ --recursive -n user-<username>
```

ArgoCD watches the `ossama` branch (`k8s/` path) and syncs automatically on every push.

The API connects to the MLflow service inside the cluster (`http://mlflow:5000`) — no local SQLite copy needed. The MLflow service itself uses S3 (`s3://<bucket>/mlruns`) as artifact backend.

### Architecture

```
GitHub push
    │
    ▼
GitHub Actions CI  ──── lint + tests
    │
    ▼
GitHub Actions CD  ──── build Docker image ──── push to Docker Hub
    │                                                   │
    │                                    update k8s/deployment.yaml
    │                                                   │
    ▼                                                   ▼
ArgoCD (SSPCloud) ──────────────────── sync Kubernetes manifests
    │
    ▼
Pod: loan-api ──── reads @champion model from ──── MLflow service (k8s)
    │                                                   │
    ├── POST /predict ──────────────────────────────────┤
    ├── GET  /metrics ── Prometheus ── Grafana          │
    └── logs/predictions.jsonl ── S3 sync ── drift_check.yml
```

---

## Monitoring

### Prometheus metrics

| Metric | Type | Description |
|--------|------|-------------|
| `loan_predictions_total{result}` | Counter | Approved / rejected counts |
| `loan_prediction_probability` | Histogram | Distribution of approval probabilities |
| `loan_prediction_errors_total` | Counter | Prediction errors |
| `loan_approval_rate` | Gauge | Rolling approval rate (last 100 predictions) |
| `loan_request_income` | Histogram | Applicant income (drift monitoring) |
| `loan_request_amount` | Histogram | Loan amount (drift monitoring) |
| `loan_request_lti_ratio` | Histogram | Loan-to-income ratio (drift monitoring) |
| `loan_batch_size` | Histogram | Batch request sizes |

### Grafana alerts (auto-provisioned)

| Alert | Condition | Severity |
|-------|-----------|----------|
| High Prediction Error Rate | > 5 errors in 5 min | Critical |
| Abnormally Low Approval Rate | < 10% for 5 min | Warning |
| High API Latency | p95 > 2s for 3 min | Warning |

---

## Configuration reference

All constants are in `config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `CV_FOLDS` | 5 | Stratified K-Fold folds during hyperparameter search |
| `MAX_EVALS` | 10 | Hyperopt iterations (increase for better results, slower training) |
| `RANDOM_STATE` | 42 | Seed for all random operations — guarantees reproducibility |
| `TEST_SIZE` | 0.2 | Holdout fraction (split into calibration + eval) |
| `F1_PROMOTION_THRESHOLD` | 0.5 | Minimum F1 required to promote a challenger to @champion |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | Overridden by env var in Kubernetes (`http://mlflow:5000`) |
| `MLFLOW_MODEL_NAME` | `loan-approval-model` | Model name in the MLflow Registry |

---

## Project checklist

| Feature | Status |
|---------|--------|
| Development best practices (pre-commit, ruff, tests) | ✅ |
| 3-way train / calibration / eval split (no leakage) | ✅ |
| Hyperparameter tuning — XGBoost, CatBoost, RandomForest | ✅ |
| Early stopping on boosted models (CV) | ✅ |
| Probability calibration (isotonic regression) | ✅ |
| Champion/challenger registry with regression guard | ✅ |
| Confusion matrix logged as MLflow artifact | ✅ |
| FastAPI — single, batch (max 500), explain, health | ✅ |
| SHAP explanations (no external shap library) | ✅ |
| Thread-safe metrics, no traceback leakage | ✅ |
| Prediction logger → S3 sync (every 10 predictions + 60s timer) | ✅ |
| Dockerfile + Docker Hub CI/CD | ✅ |
| Kubernetes deployment (SSPCloud + ArgoCD GitOps) | ✅ |
| MLflow as Kubernetes service (no ephemeral SQLite) | ✅ |
| Post-deploy healthcheck + automatic rollback | ✅ |
| Prometheus + Grafana monitoring (8 metrics, 6 panels) | ✅ |
| Grafana alerting (3 rules) | ✅ |
| Daily drift detection (KS + PSI) → auto-retraining | ✅ |
| Guard: drift check skipped if no logs yet | ✅ |
| Unit tests (28 total) | ✅ |
| Integration tests (real HTTP) | ✅ |
