"""Load the @champion model from MLflow Registry and make predictions."""

import sys
from pathlib import Path

import mlflow
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import MLFLOW_TRACKING_URI  # noqa: E402
from model.registry import load_champion_model, load_preprocessor_from_registry  # noqa: E402


def predict(data: dict) -> dict:
    """Load @champion model from registry and return a prediction.

    Args:
        data: dict with raw feature values (same format as training data, without loan_status)

    Returns:
        dict with 'loan_status' (0/1) and 'probability'
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model = load_champion_model()
    preprocessor = load_preprocessor_from_registry()

    df = pd.DataFrame([data])
    X = preprocessor.inference_transform(df)

    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    return {"loan_status": prediction, "probability": probability}


if __name__ == "__main__":
    sample = {
        "person_age": 28,
        "person_income": 45000,
        "person_home_ownership": "RENT",
        "person_emp_length": 3.0,
        "loan_intent": "PERSONAL",
        "loan_amnt": 10000,
        "loan_percent_income": 0.22,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 4,
    }

    result = predict(sample)
    print(f"Prediction: {result['loan_status']} (probability: {result['probability']:.4f})")
