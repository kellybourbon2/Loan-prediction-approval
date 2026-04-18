from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def train_model(model, X_train, y_train):

    if isinstance(model, (XGBClassifier, CatBoostClassifier)):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        if isinstance(model, XGBClassifier):
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        elif isinstance(model, CatBoostClassifier):
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

    else:
        model.fit(X_train, y_train)

    return model
