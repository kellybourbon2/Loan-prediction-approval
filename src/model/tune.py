import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def build_model(params: dict):
    print(f"Building model with params: {params}")
    model_name = params["model_name"]

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
        )

    if model_name == "xgboost":
        return xgb.XGBClassifier(
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            min_child_weight=float(params["min_child_weight"]),
            random_state=int(params["seed"]),
            n_estimators=500,  # tu peux aussi le tuner
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
    model = build_model(params)
    model.fit(X_train, y_train)
    preds = model.predict(X_train)
    f1 = f1_score(y_train, preds)
    return 1.0 - f1  # hyperopt minimise
