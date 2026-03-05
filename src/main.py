import os

import joblib
from hyperopt import Trials, fmin, space_eval, tpe

import mlflow
from config import MAX_EVALS
from data_processing.data_load import data_loading
from data_processing.preprocessing import preprocess_data
from model.evaluate import evaluate_model
from model.search_space import search_space
from model.train import train_model
from model.tune import build_model, objective

if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Loan Prediction Approval Experiments")

    df_train = data_loading(set="train")
    X_train, X_test, y_train, y_test, encoder = preprocess_data(df_train)

    trials = Trials()

    with mlflow.start_run():
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
        best_model = build_model(best_params)
        best_model = train_model(best_model, X_train, y_train)

        # Evaluate
        acc, f1, recall, precision = evaluate_model(best_model, X_test, y_test)

        # Store the best model
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        # Store the encoder as an artifact
        os.makedirs("artifacts", exist_ok=True)
        encoder_path = "artifacts/encoder.joblib"
        joblib.dump(encoder, encoder_path)
        mlflow.log_artifact(encoder_path, artifact_path="model")
