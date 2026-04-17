import logging
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from config import MLFLOW_TRACKING_URI  # noqa: E402
from model.registry import load_champion_model, load_preprocessor_from_registry  # noqa: E402
from api.schemas import (  # noqa: E402
    LoanApplication,
    PredictionResponse,
    BatchPredictionResponse,
    ExplainResponse,
)
from api.metrics import (  # noqa: E402
    PREDICTION_COUNTER,
    PREDICTION_ERRORS,
    PROBABILITY_HISTOGRAM,
    APPROVAL_RATE_GAUGE,
    INCOME_HISTOGRAM,
    LOAN_AMOUNT_HISTOGRAM,
    LTI_HISTOGRAM,
    BATCH_SIZE_HISTOGRAM,
)
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
    description="Predicts loan approval using the @champion model from MLflow Registry.",
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)  # /metrics with HTTP instrumentation

# Serve the web UI from /ui
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")


@app.get("/", include_in_schema=False)
@app.get("/ui", include_in_schema=False)
@app.get("/ui/", include_in_schema=False)
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    """Health check — verifies model and preprocessor are loaded."""
    if not hasattr(app.state, "model") or not hasattr(app.state, "preprocessor"):
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"status": "ok"}


_approval_window: list[int] = []
_approval_lock = threading.Lock()
_WINDOW_SIZE = 100


def _run_prediction(request: LoanApplication) -> PredictionResponse:
    """Core prediction logic — shared by single and batch endpoints."""
    data = request.model_dump()
    df = pd.DataFrame([data])
    X = app.state.preprocessor.inference_transform(df)
    proba_default = float(app.state.model.predict_proba(X)[0][1])
    probability = round(1.0 - proba_default, 4)
    prediction = int(probability >= 0.5)

    # Prometheus metrics
    PREDICTION_COUNTER.labels(result="approved" if prediction else "rejected").inc()
    PROBABILITY_HISTOGRAM.observe(probability)
    INCOME_HISTOGRAM.observe(data["person_income"])
    LOAN_AMOUNT_HISTOGRAM.observe(data["loan_amnt"])
    LTI_HISTOGRAM.observe(data["loan_percent_income"])

    # Rolling approval rate — thread-safe
    with _approval_lock:
        _approval_window.append(prediction)
        if len(_approval_window) > _WINDOW_SIZE:
            _approval_window.pop(0)
        APPROVAL_RATE_GAUGE.set(sum(_approval_window) / len(_approval_window))

    log_prediction(inputs=data, prediction=prediction, probability=probability)

    return PredictionResponse(
        loan_status=prediction, approved=bool(prediction), probability=probability
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: LoanApplication):
    """Predict loan approval for a single application."""
    try:
        return _run_prediction(request)
    except Exception as e:
        PREDICTION_ERRORS.inc()
        logger.exception("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Internal prediction error.") from e


_MAX_BATCH = 500


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(requests: list[LoanApplication]):
    """Predict loan approval for multiple applications in one request."""
    if not requests:
        raise HTTPException(status_code=422, detail="Request list cannot be empty.")
    if len(requests) > _MAX_BATCH:
        raise HTTPException(
            status_code=422, detail=f"Batch size exceeds maximum of {_MAX_BATCH}."
        )
    try:
        predictions = [_run_prediction(r) for r in requests]
    except Exception as e:
        PREDICTION_ERRORS.inc()
        logger.exception("Batch prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Internal prediction error.") from e

    BATCH_SIZE_HISTOGRAM.observe(len(predictions))
    approved = sum(p.approved for p in predictions)
    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        approved_count=approved,
        rejected_count=len(predictions) - approved,
        approval_rate=round(approved / len(predictions), 4),
    )


_FEATURE_LABELS = {
    "person_income": "Annual Income",
    "person_emp_length": "Employment Length",
    "loan_amnt": "Loan Amount",
    "loan_percent_income": "Loan/Income Ratio",
    "cb_person_cred_hist_length": "Credit History",
    "record_credit_default": "Prior Default",
    "age_category": "Age Group",
    "person_home_ownership": "Home Ownership",
    "loan_intent": "Loan Intent",
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _base_model(model):
    """Unwrap CalibratedClassifierCV to get the underlying estimator."""
    from sklearn.calibration import CalibratedClassifierCV

    if isinstance(model, CalibratedClassifierCV):
        return model.calibrated_classifiers_[0].estimator
    return model


def _shap_contributions(preprocessor, model, X):
    model = _base_model(model)
    model_type = type(model).__name__

    if model_type == "XGBClassifier":
        import xgboost as xgb

        dmat = xgb.DMatrix(X)
        contribs = model.get_booster().predict(dmat, pred_contribs=True)
        # shape: (1, n_features+1) — last col is bias/base in log-odds
        sv_raw = contribs[0, :-1]
        base = float(_sigmoid(contribs[0, -1]))
    elif model_type == "CatBoostClassifier":
        from catboost import Pool

        pool = Pool(X)
        contribs = model.get_feature_importance(type="ShapValues", data=pool)
        sv_raw = contribs[0, :-1]
        base = float(_sigmoid(contribs[0, -1]))
    else:
        # RandomForest fallback: global importance weighted by prediction deviation
        prob = float(model.predict_proba(X)[0][1])
        base = 0.5
        deviation = prob - base
        sv_raw = model.feature_importances_ * deviation

    # XGB/CB give log-odds contributions (positive = pushes toward default=class1)
    # Negate so positive = pushes toward approval
    sv = -np.array(sv_raw, dtype=float)

    feature_names = preprocessor.encoder.get_feature_names_out()
    cat_cols = list(preprocessor.encoder.transformers_[0][2])

    totals: dict[str, float] = {}
    for i, fname in enumerate(feature_names):
        if fname.startswith("cat__"):
            suffix = fname[5:]
            orig = next((c for c in cat_cols if suffix.startswith(c)), suffix)
        else:
            orig = fname[11:] if fname.startswith("remainder__") else fname
        totals[orig] = totals.get(orig, 0.0) + float(sv[i])

    approval_base = 1.0 - base
    return approval_base, sorted(totals.items(), key=lambda x: abs(x[1]), reverse=True)


@app.post("/explain", response_model=ExplainResponse)
def explain(request: LoanApplication):
    """Return SHAP feature contributions for a loan application."""
    try:
        df = pd.DataFrame([request.model_dump()])
        X = app.state.preprocessor.inference_transform(df)
        base, contributions = _shap_contributions(
            app.state.preprocessor, app.state.model, X
        )
    except Exception as e:
        logger.exception("Explain error: %s", e)
        raise HTTPException(
            status_code=500, detail="Internal explanation error."
        ) from e

    return ExplainResponse(
        base_value=round(base, 4),
        features=[
            {"feature": f, "label": _FEATURE_LABELS.get(f, f), "shap": round(v, 4)}
            for f, v in contributions
        ],
    )
