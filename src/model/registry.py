import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import F1_PROMOTION_THRESHOLD, MLFLOW_MODEL_NAME  # noqa: E402


def register_model(run_id: str, model_name: str = MLFLOW_MODEL_NAME) -> str:
    """Register the model from a run into the MLflow Model Registry.

    Returns the registered model version number.
    """
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)

    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias="challenger",
        version=mv.version,
    )
    print(f"Model v{mv.version} registered as '{model_name}' with alias @challenger")
    return mv.version


def promote_to_champion(
    version: str,
    f1: float,
    model_name: str = MLFLOW_MODEL_NAME,
    threshold: float = F1_PROMOTION_THRESHOLD,
) -> bool:
    """Promote model version to @champion if F1 exceeds threshold.

    Returns True if promoted, False otherwise.
    """
    client = MlflowClient()

    if f1 >= threshold:
        # Remove champion alias from previous version if it exists
        try:
            old_champion = client.get_model_version_by_alias(model_name, "champion")
            client.delete_registered_model_alias(name=model_name, alias="champion")
            print(f"Removed @champion from previous version v{old_champion.version}")
        except mlflow.exceptions.MlflowException:
            pass  # no previous champion alias exists yet

        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=version,
        )
        print(f"Model v{version} promoted to @champion (F1={f1:.4f} >= {threshold})")
        return True

    print(
        f"Model v{version} NOT promoted (F1={f1:.4f} < threshold={threshold}). "
        "Stays as @challenger."
    )
    return False


def load_champion_model(model_name: str = MLFLOW_MODEL_NAME):
    """Load the @champion model from the MLflow Model Registry."""
    model_uri = f"models:/{model_name}@champion"
    return mlflow.sklearn.load_model(model_uri)


def get_champion_run_id(model_name: str = MLFLOW_MODEL_NAME) -> str:
    """Return the run_id associated with the current @champion model version."""
    client = MlflowClient()
    mv = client.get_model_version_by_alias(model_name, "champion")
    return mv.run_id


def load_preprocessor_from_registry(model_name: str = MLFLOW_MODEL_NAME):
    """Download and load the fitted DataPreprocessor from the @champion run artifacts."""
    import joblib
    import tempfile

    client = MlflowClient()
    run_id = get_champion_run_id(model_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = client.download_artifacts(
            run_id=run_id,
            path="model/preprocessor.joblib",
            dst_path=tmpdir,
        )
        preprocessor = joblib.load(local_path)

    return preprocessor
