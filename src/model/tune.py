import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from config import CV_FOLDS, RANDOM_STATE


def build_model(params: dict, y_train=None):
    print(f"Building model with params: {params}")
    model_name = params["model_name"]

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )

    if model_name == "xgboost":
        scale_pos_weight = 1.0
        if y_train is not None:
            y_arr = np.array(y_train)
            neg, pos = (y_arr == 0).sum(), (y_arr == 1).sum()
            if pos > 0:
                scale_pos_weight = neg / pos
        return xgb.XGBClassifier(
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            min_child_weight=float(params["min_child_weight"]),
            random_state=int(params["seed"]),
            n_estimators=500,
            early_stopping_rounds=50,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
        )

    if model_name == "catboost":
        return CatBoostClassifier(
            depth=int(params["depth"]),
            learning_rate=float(params["learning_rate"]),
            l2_leaf_reg=float(params["l2_leaf_reg"]),
            iterations=int(params["iterations"]),
            early_stopping_rounds=50,
            random_seed=int(params["seed"]),
            auto_class_weights="Balanced",
            verbose=False,
        )

    raise ValueError(f"Unknown model_name: {model_name}")


def objective(params, X_train, y_train):
    """Evaluate model using stratified cross-validation — avoids overfitting to train set."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    boosted = params["model_name"] in ("xgboost", "catboost")

    f1_scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = np.array(y_train)[train_idx], np.array(y_train)[val_idx]
        model = build_model(params, y_tr)
        if boosted:
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_tr, y_tr)
        f1_scores.append(f1_score(y_val, model.predict(X_val)))

    return 1.0 - np.mean(f1_scores)  # hyperopt minimises
