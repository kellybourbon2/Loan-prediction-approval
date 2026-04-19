import os
import sys
import tempfile
from pathlib import Path

import joblib
from dotenv import load_dotenv
from hyperopt import Trials, fmin, space_eval, tpe
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split

import mlflow

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)  # so can import config, data_processing


from config import MAX_EVALS, MLFLOW_EXPERIMENT_NAME, MLFLOW_MODEL_NAME  # noqa: E402

from data_processing.data_load import data_loading  # noqa: E402
from data_processing.preprocessing import preprocess_data  # noqa: E402
from model.evaluate import evaluate_model  # noqa: E402
from model.registry import promote_to_champion, register_model  # noqa: E402
from model.search_space import search_space  # noqa: E402
from model.train import train_model  # noqa: E402
from model.tune import build_model, objective  # noqa: E402

load_dotenv(override=True)  # override default mlflow variables with .env variables

if __name__ == "__main__":
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df_train = data_loading(set="train")
    X_train, X_holdout, y_train, y_holdout, preprocessor = preprocess_data(df_train)

    # Split holdout into calibration (50 %) + eval (50 %) — never seen during training
    X_cal, X_eval, y_cal, y_eval = train_test_split(
        X_holdout, y_holdout, test_size=0.5, random_state=42
    )

    trials = Trials()

    with mlflow.start_run() as run:
        best = fmin(
            fn=lambda params: objective(params, X_train, y_train),
            space=search_space,
            algo=tpe.suggest,
            max_evals=MAX_EVALS,
            trials=trials,
        )
        # model_name + hyperparams
        best_params = space_eval(search_space, best)
        model_name = best_params["model_name"]

        # Log model name + params
        mlflow.log_param("best_model_name", model_name)
        for k, v in best_params.items():
            if k != "model_name":
                mlflow.log_param(f"best_{k}", v)

        # Rebuild + train BEST model
        best_model = build_model(best_params, y_train)
        best_model = train_model(best_model, X_train, y_train)

        # Calibrate probabilities on the calibration split (never seen during training)
        calibrated_model = CalibratedClassifierCV(
            FrozenEstimator(best_model), method="isotonic"
        )

        calibrated_model.fit(X_cal, y_cal)
        mlflow.log_param("calibration_method", "isotonic")

        # Evaluate on the held-out eval split (not the calibration set)
        acc, f1, recall, precision = evaluate_model(calibrated_model, X_eval, y_eval)

        # Store the full preprocessor as an artifact temporary
        # with tempfile to avoid polluting repo (scaler + encoder + ordinal encoder)
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessor_path = os.path.join(tmpdir, "preprocessor.joblib")
            joblib.dump(preprocessor, preprocessor_path)
            mlflow.log_artifact(preprocessor_path, artifact_path="model")
            # Log calibrated model + preprocessor in the same artifact folder
            mlflow.sklearn.log_model(calibrated_model, artifact_path="model")
            mlflow.log_artifact(preprocessor_path, artifact_path="model")

        run_id = run.info.run_id

    # Register model in MLflow Model Registry
    version = register_model(run_id=run_id, model_name=MLFLOW_MODEL_NAME)

    # Promote to @champion if F1 is good enough
    promote_to_champion(version=version, f1=f1, model_name=MLFLOW_MODEL_NAME)
