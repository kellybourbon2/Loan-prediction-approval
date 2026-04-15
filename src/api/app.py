import sys
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from config import MLFLOW_TRACKING_URI  # noqa: E402
from model.registry import load_champion_model, load_preprocessor_from_registry  # noqa: E402
from api.schemas import LoanApplication, PredictionResponse  # noqa: E402
from api.metrics import PREDICTION_COUNTER, PREDICTION_ERRORS, PROBABILITY_HISTOGRAM  # noqa: E402
from api.logger import log_prediction  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and preprocessor once at startup, release on shutdown."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        app.state.model = load_champion_model()
        app.state.preprocessor = load_preprocessor_from_registry()
    except Exception as e:
        raise RuntimeError(f"Failed to load model from MLflow Registry: {e}") from e
    yield


app = FastAPI(
    title="Loan Approval Prediction API",
    description="Predicts whether a loan application will be approved using the @champion model from MLflow Registry.",
    version="1.0.0",
    lifespan=lifespan,
)

# Expose /metrics endpoint with automatic HTTP instrumentation (latency, request count, status codes)
Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: LoanApplication):
    """Predict loan approval for a given application."""
    try:
        df = pd.DataFrame([request.model_dump()])
        X = app.state.preprocessor.inference_transform(df)
        proba_default = float(app.state.model.predict_proba(X)[0][1])
        probability = round(
            1.0 - proba_default, 4
        )  # probability of approval (no default)
        prediction = int(probability >= 0.5)
    except Exception as e:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Update Prometheus metrics
    PREDICTION_COUNTER.labels(result="approved" if prediction else "rejected").inc()
    PROBABILITY_HISTOGRAM.observe(probability)

    # Structured log for drift analysis
    log_prediction(
        inputs=request.model_dump(), prediction=prediction, probability=probability
    )

    return PredictionResponse(
        loan_status=prediction,
        approved=bool(prediction),
        probability=probability,
    )
