# Loan-prediction
Project part of Ensae "Mise en production" course — Parcours MLOps

## Architecture MLOps

![MLOps Architecture](mlops.png)

# How to start

Clone the repo and follow the steps below.

## Environment setup

```bash
uv sync
```

## Pre-commit configuration

```bash
pre-commit install
```

## SSPCloud credentials

Data is stored on **SSPCloud** (MinIO object storage).
Download your credentials from **[datalab.sspcloud.fr](https://datalab.sspcloud.fr) → My account → Storage access** and create a `.env` file at the root of the project:

```bash
cp .env.example .env
# then fill in your SSPCloud credentials
```

`.env.example`:
```
AWS_ACCESS_KEY_ID=<your_sspcloud_access_key>
AWS_SECRET_ACCESS_KEY=<your_sspcloud_secret_key>
AWS_SESSION_TOKEN=<your_sspcloud_session_token>
AWS_S3_ENDPOINT=minio.lab.sspcloud.fr
AWS_BUCKET_NAME=<your_bucket_name>
```

The dataset (Kaggle Playground S4E10) must be uploaded to your SSPCloud bucket as `train.csv`.

## Train the model

```bash
uv run python src/main.py
```

This will:
1. Load data from SSPCloud
2. Preprocess and train the model (Hyperopt tuning)
3. Log the run to MLflow and register the model in the registry
4. Promote to `@champion` if F1 ≥ 0.5

## MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open: http://127.0.0.1:5000

## Run the API

```bash
uvicorn src.api.app:app --reload --port 8000
```

Endpoints:
- `GET /health` — health check
- `POST /predict` — loan approval prediction
- `GET /metrics` — Prometheus metrics
- `GET /docs` — Swagger UI

## Run the full stack (API + Prometheus + Grafana)

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |

## Drift analysis

```bash
uv run python src/drift_analysis.py
```

## Run tests

```bash
uv run pytest unit_tests/ -v
```
