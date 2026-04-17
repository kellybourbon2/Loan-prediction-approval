import mlflow
from mlflow import MlflowClient

from config import F1_PROMOTION_THRESHOLD, MLFLOW_MODEL_NAME


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


def _get_champion_f1(model_name: str, client: MlflowClient) -> float | None:
    """Return the test_f1_score of the current @champion, or None if no champion."""
    try:
        mv = client.get_model_version_by_alias(model_name, "champion")
        run = client.get_run(mv.run_id)
        return run.data.metrics.get("test_f1_score")
    except mlflow.exceptions.MlflowException:
        return None


def promote_to_champion(
    version: str,
    f1: float,
    model_name: str = MLFLOW_MODEL_NAME,
    threshold: float = F1_PROMOTION_THRESHOLD,
) -> bool:
    """Promote model version to @champion only if:
      1. F1 >= absolute threshold
      2. F1 > current champion's F1 (regression guard)

    Returns True if promoted, False otherwise.
    """
    client = MlflowClient()

    if f1 < threshold:
        print(
            f"Model v{version} NOT promoted (F1={f1:.4f} < threshold={threshold}). "
            "Stays as @challenger."
        )
        return False

    champion_f1 = _get_champion_f1(model_name, client)
    if champion_f1 is not None and f1 <= champion_f1:
        print(
            f"Model v{version} NOT promoted — regression detected "
            f"(challenger F1={f1:.4f} <= champion F1={champion_f1:.4f})."
        )
        return False

    # Remove champion alias from previous version
    try:
        old_champion = client.get_model_version_by_alias(model_name, "champion")
        client.delete_registered_model_alias(name=model_name, alias="champion")
        print(f"Removed @champion from previous version v{old_champion.version}")
    except mlflow.exceptions.MlflowException:
        pass

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=version,
    )
    if champion_f1 is not None:
        print(
            f"Model v{version} promoted to @champion "
            f"(F1={f1:.4f} > previous champion F1={champion_f1:.4f})"
        )
    else:
        print(f"Model v{version} promoted to @champion (F1={f1:.4f}, first champion)")
    return True


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
