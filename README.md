# Loan Prediction Approval — MLOps Project

> ENSAE Paris — *Mise en production* course | Parcours MLOps

A full MLOps pipeline for predicting loan approval, covering data processing, model training with hyperparameter tuning, MLflow experiment tracking, FastAPI deployment, Kubernetes orchestration, GitOps automation, Grafana/Prometheus monitoring, XAI explanations, and automated drift-triggered retraining.

---

## Live URLs (SSPCloud — namespace `user-oualy`)

| Service | URL |
|---------|-----|
| **Web UI** | https://loan-api-oualy.user.lab.sspcloud.fr |
| **Swagger UI** | https://loan-api-oualy.user.lab.sspcloud.fr/docs |
| **Prometheus metrics** | https://loan-api-oualy.user.lab.sspcloud.fr/metrics |
| **Grafana dashboard** | https://grafana-loan-oualy.user.lab.sspcloud.fr (admin/admin) |
| **MLflow UI** | https://mlflow-oualy.user.lab.sspcloud.fr |

---

## Project Checklist

| Step | Status |
|------|--------|
| Development best practices (pre-commit, linting, tests) | ✅ |
| ML model for a business need (loan approval) | ✅ |
| Cross-validation + hyperparameter fine-tuning | ✅ |
| Probability calibration (isotonic regression) | ✅ |
| Reproducible fine-tuning via MLflow | ✅ |
| FastAPI to expose the best model | ✅ |
| Batch prediction endpoint | ✅ |
| XAI — SHAP feature explanations | ✅ |
| Dockerfile + Docker Hub CI/CD | ✅ |
| Deploy on SSP Cloud (Kubernetes + ArgoCD) | ✅ |
| GitOps continuous deployment | ✅ |
| Monitoring (Prometheus + Grafana) | ✅ |
| Grafana alerting (errors, latency, approval rate) | ✅ |
| Drift detection → automatic retraining | ✅ |
| Model regression guard before promotion | ✅ |
| Unit tests — preprocessing + API endpoints (14 tests) | ✅ |

---

## Repository Structure

```
├── src/
│   ├── api/
│   │   ├── app.py            # FastAPI app (predict, batch, explain endpoints)
│   │   ├── schemas.py        # Pydantic schemas
│   │   ├── metrics.py        # Prometheus metrics
│   │   └── logger.py         # Structured prediction logger (+ S3 sync)
│   ├── model/
│   │   ├── train.py          # Model training
│   │   ├── tune.py           # Hyperopt tuning
│   │   ├── evaluate.py       # Metrics evaluation
│   │   ├── registry.py       # MLflow registry + champion promotion
│   │   └── search_space.py   # Hyperopt search space
│   ├── data_processing/
│   │   ├── preprocessing.py  # DataPreprocessor (clean, encode, scale)
│   │   └── data_load.py      # S3 data loading
│   ├── main.py               # Full training entrypoint
│   └── drift_analysis.py     # Feature & prediction drift detection
├── .github/workflows/
│   ├── ci.yml                # Lint + unit tests on every push
│   ├── cd.yml                # Build & push Docker image (src changes only)
│   ├── retrain.yml           # Manual / scheduled retraining
│   └── drift_check.yml       # Daily drift check → triggers retrain if needed
├── k8s/                      # Kubernetes manifests (ArgoCD GitOps)
├── monitoring/
│   ├── prometheus.yml        # Scrape config
│   └── grafana/
│       ├── dashboards/       # Dashboard JSON (auto-provisioned)
│       └── provisioning/
│           ├── datasources/  # Prometheus datasource
│           ├── dashboards/   # Dashboard provisioning
│           └── alerting/     # Alert rules + contact points
├── unit_tests/
│   ├── test_preprocessing.py # DataPreprocessor tests
│   └── test_api.py           # API endpoint tests (predict, batch, explain)
├── Dockerfile
├── docker-compose.yml        # Local stack (API + Prometheus + Grafana)
├── pyproject.toml
└── config.py
```

---

## 1 — Environment Setup

**Requirements:** Python ≥ 3.13, [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/kellybourbon2/Loan-prediction-approval.git
cd Loan-prediction-approval
git checkout ossama
uv sync
```

### SSPCloud credentials

```bash
cp .env.example .env
# Fill in your values
```

```env
AWS_ACCESS_KEY_ID=<your_key>
AWS_SECRET_ACCESS_KEY=<your_secret>
AWS_SESSION_TOKEN=<your_token>
AWS_S3_ENDPOINT=minio.lab.sspcloud.fr
AWS_BUCKET_NAME=<your_bucket>
```

The dataset (Kaggle Playground S4E10) must be uploaded to your bucket as `train.csv`.

---

## 2 — Development Best Practices

```bash
# Linting
uv run ruff check src/
uv run black src/

# Unit tests (14 tests: preprocessing + API endpoints)
uv run pytest unit_tests/ -v
```

Tests cover:
- **Preprocessing:** clean, feature engineering, split, encoding
- **API `/predict`:** valid payload, approval/rejection logic, invalid inputs (422)
- **API `/predict/batch`:** structure, counts, approval rate, empty list guard
- **API `/explain`:** feature structure, SHAP values range, base value

---

## 3 — Model Training (MLflow)

```bash
uv run python src/main.py
```

Pipeline:
1. Load `train.csv` from SSPCloud (MinIO)
2. Preprocess (`DataPreprocessor`: clean → feature engineering → scale → encode)
3. 5-fold cross-validation + hyperparameter tuning with **Hyperopt** (`MAX_EVALS=10`)
4. Train best model (XGBoost / CatBoost / RandomForest)
5. **Calibrate probabilities** with `CalibratedClassifierCV(method='isotonic')` on a held-out calibration split
6. Evaluate on a separate eval split
7. Log run to **MLflow** (metrics, params, artifacts)
8. Register in MLflow Model Registry as `@challenger`
9. **Promote to `@champion`** only if:
   - F1 ≥ `F1_PROMOTION_THRESHOLD` (0.5)
   - F1 > current champion's F1 (regression guard — new model must be strictly better)

```bash
# Inspect experiments
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
# → http://127.0.0.1:5000
```

---

## 4 — FastAPI

```bash
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI (loan assessment form) |
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Single loan approval prediction |
| `POST` | `/predict/batch` | Batch prediction (list of applications) |
| `POST` | `/explain` | SHAP feature contributions for one application |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/docs` | Swagger UI |

### Single prediction

```bash
curl -X POST https://loan-api-oualy.user.lab.sspcloud.fr/predict \
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

### Batch prediction

```bash
curl -X POST https://loan-api-oualy.user.lab.sspcloud.fr/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{...}, {...}, {...}]'
# → {"predictions": [...], "total": 3, "approved_count": 2, "rejected_count": 1, "approval_rate": 0.667}
```

### XAI — SHAP explanation

```bash
curl -X POST https://loan-api-oualy.user.lab.sspcloud.fr/explain \
  -H "Content-Type: application/json" \
  -d '{...same payload as /predict...}'
# → {
#     "base_value": 0.62,
#     "features": [
#       {"feature": "loan_percent_income", "label": "Loan/Income Ratio", "shap": -0.183},
#       {"feature": "person_income",       "label": "Annual Income",     "shap": +0.091},
#       ...
#     ]
#   }
```

SHAP values are computed using native tree contributions (`pred_contribs=True` for XGBoost, `ShapValues` for CatBoost) — no external `shap` library required. Positive values push toward approval, negative toward rejection.

---

## 5 — Docker

```bash
# Full local stack (API + Prometheus + Grafana)
docker compose up
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |

The Docker image is automatically built and pushed to Docker Hub (`oualyoss/loan-api`) by the CD workflow on every push that modifies `src/`, `Dockerfile`, `pyproject.toml`, or `uv.lock`.

---

## 6 — CI/CD Workflows

| Workflow | Trigger | Action |
|----------|---------|--------|
| `ci.yml` | Every push | Lint (Ruff) + unit tests |
| `cd.yml` | Push on `src/**`, `Dockerfile`, `pyproject.toml` | Build Docker image → push to Docker Hub → update `k8s/deployment.yaml` |
| `retrain.yml` | Manual / every Monday 2am UTC | Full retraining + MLflow registry update |
| `drift_check.yml` | Daily 8am UTC | Download logs from S3 → drift analysis → trigger `retrain.yml` if drift detected |

---

## 7 — Drift Detection & Automatic Retraining

Prediction logs are written to `logs/predictions.jsonl` and synced to S3 every 100 predictions.

The `drift_check.yml` workflow runs daily:
1. Downloads logs from S3
2. Runs `src/drift_analysis.py` (KS test + PSI for numerical features, distribution shift for categorical)
3. If drift detected on any feature → triggers `retrain.yml` automatically

```bash
# Run drift analysis manually
uv run python src/drift_analysis.py --log-file logs/predictions.jsonl

# Exit with code 1 if drift detected (used in CI)
uv run python src/drift_analysis.py --fail-on-drift
```

---

## 8 — Kubernetes Deployment (SSPCloud)

### One-time setup

```bash
# SSPCloud credentials
kubectl create secret generic loan-api-secret \
  --from-literal=AWS_ACCESS_KEY_ID=<key> \
  --from-literal=AWS_SECRET_ACCESS_KEY=<secret> \
  --from-literal=AWS_SESSION_TOKEN=<token> \
  --from-literal=AWS_S3_ENDPOINT=https://minio.lab.sspcloud.fr \
  --from-literal=AWS_BUCKET_NAME=oualy \
  -n user-oualy

# MLflow database
kubectl create secret generic mlflow-db --from-file=mlflow.db=mlflow.db -n user-oualy
```

### Deploy

```bash
kubectl apply -f k8s/ --recursive -n user-oualy
```

ArgoCD watches the `ossama` branch (`k8s/` path) and auto-syncs on every push.

---

## 9 — Monitoring

### Prometheus metrics

| Metric | Type | Description |
|--------|------|-------------|
| `loan_predictions_total{result}` | Counter | Predictions by outcome (approved/rejected) |
| `loan_prediction_probability` | Histogram | Distribution of approval probabilities |
| `loan_prediction_errors_total` | Counter | Prediction errors |
| `loan_approval_rate` | Gauge | Rolling approval rate (last 100 predictions) |
| `loan_request_income` | Histogram | Applicant income distribution (drift detection) |
| `loan_request_amount` | Histogram | Loan amount distribution (drift detection) |
| `loan_request_lti_ratio` | Histogram | Loan-to-income ratio distribution (drift detection) |
| `loan_batch_size` | Histogram | Batch request sizes |

### Grafana dashboard

Dashboard **"Loan Approval API"** at https://grafana-loan-oualy.user.lab.sspcloud.fr:

- Request rate, approval rate, predictions approved vs rejected
- Prediction probability distribution (p50/p90/p99)
- API latency (p50/p95), prediction errors
- Rolling approval rate gauge
- Income / loan amount / LTI ratio distributions (data drift monitoring)
- Batch request sizes

### Grafana alerting

Three alert rules provisioned automatically:

| Alert | Condition | Severity |
|-------|-----------|----------|
| High Prediction Error Rate | > 5 errors in 5 min | Critical |
| Abnormally Low Approval Rate | < 10% for 5 min | Warning |
| High API Latency | p95 > 2s for 3 min | Warning |

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CV_FOLDS` | 5 | Cross-validation folds |
| `MAX_EVALS` | 10 | Hyperopt iterations |
| `F1_PROMOTION_THRESHOLD` | 0.5 | Min F1 to promote to @champion |
| `MLFLOW_MODEL_NAME` | `loan-approval-model` | Registry model name |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | Override via env var |
