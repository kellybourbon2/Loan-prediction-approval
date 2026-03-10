import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import CV_FOLDS, RANDOM_STATE  # noqa: E402


def build_model(params: dict):
    print(f"Building model with params: {params}")
    model_name = params["model_name"]

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            random_state=RANDOM_STATE,
        )

    if model_name == "xgboost":
        return xgb.XGBClassifier(
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            min_child_weight=float(params["min_child_weight"]),
            random_state=int(params["seed"]),
            n_estimators=500,
            eval_metric="logloss",
            n_jobs=-1,
        )

    if model_name == "catboost":
        return CatBoostClassifier(
            depth=int(params["depth"]),
            learning_rate=float(params["learning_rate"]),
            l2_leaf_reg=float(params["l2_leaf_reg"]),
            iterations=int(params["iterations"]),
            random_seed=int(params["seed"]),
            verbose=False,
        )

    raise ValueError(f"Unknown model_name: {model_name}")


def objective(params, X_train, y_train):
    """Evaluate model using stratified cross-validation — avoids overfitting to train set."""
    model = build_model(params)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    f1_scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = np.array(y_train)[train_idx], np.array(y_train)[val_idx]
        model.fit(X_tr, y_tr)
        f1_scores.append(f1_score(y_val, model.predict(X_val)))

    return 1.0 - np.mean(f1_scores)  # hyperopt minimises
