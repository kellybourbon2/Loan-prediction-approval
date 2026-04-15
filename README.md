# Loan Prediction Approval — MLOps Project

> ENSAE Paris — *Mise en production* course | Parcours MLOps

A full MLOps pipeline for predicting loan approval, covering data processing, model training with hyperparameter tuning, MLflow experiment tracking, FastAPI deployment, Kubernetes orchestration, GitOps automation, and Grafana/Prometheus monitoring.

![MLOps Architecture](mlops.png)

---

## Live URLs (SSPCloud — namespace `user-oualy`)

| Service | URL |
|---------|-----|
| **API** | https://loan-api-oualy.user.lab.sspcloud.fr |
| **Swagger UI** | https://loan-api-oualy.user.lab.sspcloud.fr/docs |
| **Prometheus metrics** | https://loan-api-oualy.user.lab.sspcloud.fr/metrics |
| **Grafana dashboard** | https://grafana-loan-oualy.user.lab.sspcloud.fr (admin/admin) |
| **ArgoCD UI** | https://argocd-oualy.user.lab.sspcloud.fr (admin) |

---

## Project Checklist

| Step | Status |
|------|--------|
| Development best practices (pre-commit, linting, tests) | ✅ |
| ML model for a business need (loan approval) | ✅ |
| Cross-validation + hyperparameter fine-tuning | ✅ |
| Reproducible fine-tuning via MLflow | ✅ |
| FastAPI to expose the best model | ✅ |
| Dockerfile | ✅ |
| Deploy on SSP Cloud | ✅ |
| GitOps continuous deployment | ✅ |
| Monitoring (Prometheus + Grafana) | ✅ |

---

## Repository Structure

```
├── src/
│   ├── api/              # FastAPI app (app.py, schemas, metrics, logger)
│   ├── model/            # Training, tuning, evaluation, MLflow registry
│   ├── data_processing/  # Preprocessing pipeline
│   ├── main.py           # Full training entrypoint
│   └── drift_analysis.py # Feature drift detection
├── k8s/                  # Kubernetes manifests
│   ├── deployment.yaml   # loan-api Deployment
│   ├── service.yaml      # ClusterIP services
│   ├── ingress.yaml      # Public ingress (SSPCloud)
│   ├── prometheus.yaml   # Prometheus ConfigMap + Deployment
│   ├── grafana.yaml      # Grafana ConfigMap + Deployment + dashboard
│   ├── mlflow.yaml       # MLflow tracking server
│   ├── pvc.yaml          # PersistentVolumeClaim for MLflow
│   ├── secret.yaml       # SSPCloud credentials (not in Git)
│   ├── gitops-sync.yaml  # GitOps CronJob controller
│   └── argocd-app.yaml   # ArgoCD Application manifest
├── monitoring/
│   ├── prometheus.yml    # Scrape config
│   └── grafana/          # Provisioning + dashboard JSON
├── unit_tests/           # Pytest test suite
├── Dockerfile            # Production image (uv-based)
├── docker-compose.yml    # Local full-stack (API + Prometheus + Grafana)
├── pyproject.toml        # Dependencies (uv)
└── config.py             # Global constants
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

Download your credentials from [datalab.sspcloud.fr](https://datalab.sspcloud.fr) → *My account → Storage access* and create a `.env` file:

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
MLFLOW_S3_ENDPOINT_URL=https://minio.lab.sspcloud.fr
```

The dataset (Kaggle Playground S4E10) must be uploaded to your bucket as `train.csv`.

### Pre-commit hooks

```bash
pre-commit install
```

---

## 2 — Development Best Practices

### Linting & formatting

```bash
uv run ruff check src/
uv run black src/
uv run isort src/
uv run flake8 src/
```

### Unit tests

```bash
uv run pytest unit_tests/ -v
```

Tests cover the preprocessing pipeline (`DataPreprocessor`): fit/transform contract, column types, OHE encoding, and age binning.

---

## 3 — Model Training (MLflow)

### Train and register the model

```bash
uv run python src/main.py
```

This pipeline:
1. Loads `train.csv` from SSPCloud (MinIO)
2. Preprocesses data (`DataPreprocessor`)
3. Runs 5-fold cross-validation
4. Fine-tunes hyperparameters with **Hyperopt** (`MAX_EVALS=10`)
5. Logs the run to **MLflow** (metrics, params, artifacts)
6. Registers the model in the **MLflow Model Registry**
7. Promotes to `@champion` alias if F1 ≥ 0.5

Algorithms evaluated: XGBoost, LightGBM, CatBoost, Scikit-learn (LogisticRegression, RandomForest, GradientBoosting).

### Inspect experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → http://127.0.0.1:5000
```

### Current champion

- **Model:** `loan-approval-model` — version 4 — alias `@champion`
- **Artifacts:** stored on SSPCloud MinIO (`s3://oualy/mlruns/`)

---

## 4 — FastAPI

### Run locally

```bash
uv run uvicorn src.api.app:app --reload --port 8000
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Loan approval prediction |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/docs` | Swagger UI |

### Example prediction

```bash
curl -X POST https://loan-api-oualy.user.lab.sspcloud.fr/predict \
  -H "Content-Type: application/json" \
  -d '{
    "person_age": 30,
    "person_income": 50000,
    "person_emp_exp": 5,
    "person_emp_length": 5,
    "loan_amnt": 10000,
    "loan_int_rate": 10.5,
    "loan_percent_income": 0.2,
    "cb_person_cred_hist_length": 5,
    "credit_score": 700,
    "person_gender": "male",
    "person_education": "Bachelor",
    "person_home_ownership": "RENT",
    "loan_intent": "PERSONAL",
    "loan_grade": "B",
    "cb_person_default_on_file": "N"
  }'
# → {"loan_status": 1, "approved": true, "probability": 0.9032}
```

---

## 5 — Docker

### Build & run locally

```bash
docker build -t loan-api .
docker run -p 8000:8000 \
  --env-file .env \
  -e MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
  loan-api
```

### Full local stack (API + Prometheus + Grafana)

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |

> **Note:** Docker is not available on SSPCloud shared infrastructure. The Dockerfile is production-ready and builds correctly on any standard machine. On SSPCloud, the pod clones the repo and installs dependencies at runtime via `uv sync`.

---

## 6 — Kubernetes Deployment (SSPCloud)

All manifests are in `k8s/`. The namespace is `user-oualy`.

### Prerequisites

```bash
# kubectl configured with rights on user-oualy namespace
kubectl get pods -n user-oualy
```

### Create the secrets (one-time setup)

```bash
# SSPCloud credentials secret
kubectl create secret generic loan-api-secret \
  --from-literal=AWS_ACCESS_KEY_ID=<key> \
  --from-literal=AWS_SECRET_ACCESS_KEY=<secret> \
  --from-literal=AWS_SESSION_TOKEN=<token> \
  --from-literal=AWS_S3_ENDPOINT=https://minio.lab.sspcloud.fr \
  --from-literal=AWS_DEFAULT_REGION=us-east-1 \
  --from-literal=MLFLOW_S3_ENDPOINT_URL=https://minio.lab.sspcloud.fr \
  --from-literal=AWS_BUCKET_NAME=oualy \
  -n user-oualy

# MLflow database secret
kubectl create secret generic mlflow-db \
  --from-file=mlflow.db=mlflow.db \
  -n user-oualy
```

### Deploy the full stack

```bash
kubectl apply -f k8s/prometheus.yaml
kubectl apply -f k8s/grafana.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/mlflow.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/gitops-sync.yaml
```

### Check status

```bash
kubectl get pods -n user-oualy
kubectl get ingress -n user-oualy
```

### Architecture on SSPCloud

The API pod:
1. Clones the repo from GitHub (branch `ossama`) at startup
2. Runs `uv sync --frozen` to install dependencies
3. Loads the `@champion` model from MLflow Registry (artifacts on MinIO)
4. Starts `uvicorn` on port 8000

---

## 7 — GitOps Continuous Deployment

Deployment is automated via a custom GitOps controller (`k8s/gitops-sync.yaml`) — a Kubernetes CronJob running every 2 minutes that:

1. Queries the GitHub API for the latest commit SHA on branch `ossama`
2. Compares with the last deployed SHA (stored in ConfigMap `gitops-state`)
3. If different: clones the repo and runs `kubectl apply` on all manifests
4. Updates the state ConfigMap with the new SHA

### Workflow

```
git push origin ossama
      ↓ (within 2 minutes)
gitops-sync CronJob detects new SHA
      ↓
kubectl apply -f k8s/*.yaml
      ↓
Cluster updated automatically
```

### Monitor sync status

```bash
# See deployed SHA and last sync time
kubectl get configmap gitops-state -n user-oualy -o jsonpath='{.data}'

# See recent sync jobs
kubectl get jobs -n user-oualy | grep gitops

# Logs of the last sync
kubectl logs -n user-oualy -l job-name=<job-name> --tail=50
```

### ArgoCD (UI only)

An ArgoCD instance is deployed at https://argocd-oualy.user.lab.sspcloud.fr. The `Application` manifest (`k8s/argocd-app.yaml`) is configured to watch this repository. On SSPCloud shared infrastructure, ArgoCD's application controller requires cluster-level RBAC that is not available to users; the `gitops-sync` CronJob provides equivalent GitOps automation with namespace-scoped permissions only.

```yaml
# k8s/argocd-app.yaml — paste in ArgoCD UI → New App → Edit as YAML
# on a cluster with full ArgoCD admin access
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: loan-prediction-api
spec:
  project: default
  source:
    repoURL: https://github.com/kellybourbon2/Loan-prediction-approval
    targetRevision: ossama
    path: k8s
    directory:
      exclude: "argocd-app.yaml,secret.yaml"
  destination:
    server: https://kubernetes.default.svc
    namespace: user-oualy
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

---

## 8 — Monitoring

### Prometheus

Scrapes `/metrics` on `loan-api:80` every 15 seconds.

Custom metrics exposed by the API:
- `loan_predictions_total{result}` — prediction count (approved/rejected)
- `loan_prediction_probability` — histogram of approval probabilities
- `loan_prediction_errors_total` — prediction errors

### Grafana

Dashboard **"Loan Approval API"** available at https://grafana-loan-oualy.user.lab.sspcloud.fr (admin/admin).

Panels:
- Request Rate (req/s)
- Approval Rate (%)
- Predictions (approved vs rejected)
- Prediction Probability Distribution (p50/p90/p99)
- API Latency (p50/p95)
- Prediction Errors

### Drift analysis

```bash
uv run python src/drift_analysis.py
```

---

## 9 — Re-run Everything From Scratch

```bash
# 1. Clone & setup
git clone https://github.com/kellybourbon2/Loan-prediction-approval.git
cd Loan-prediction-approval && git checkout ossama
uv sync && pre-commit install

# 2. Configure credentials
cp .env.example .env  # fill in SSPCloud credentials

# 3. Train the model
uv run python src/main.py

# 4. Test locally
uv run uvicorn src.api.app:app --reload
uv run pytest unit_tests/ -v

# 5. On Kubernetes (SSPCloud)
# → Create secrets (see section 6)
# → kubectl apply -f k8s/  (except secret.yaml and argocd-app.yaml)

# 6. GitOps active: any push to ossama branch
#    triggers automatic redeployment within 2 minutes
git add k8s/
git commit -m "update: ..."
git push origin ossama
```

---

## Configuration

Key constants in `config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `CV_FOLDS` | 5 | Cross-validation folds |
| `MAX_EVALS` | 10 | Hyperopt iterations |
| `F1_PROMOTION_THRESHOLD` | 0.5 | Min F1 to promote to @champion |
| `MLFLOW_MODEL_NAME` | `loan-approval-model` | Registry model name |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | Override via env var |

