import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from config import COLUMNS_TO_DROP, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def preprocess_data(
    df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
):

    # drop the columns that are not useful for the model
    df = df.drop(columns=COLUMNS_TO_DROP)

    # Separate target
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Identify categorical columns
    categorical_cols = X_train.select_dtypes(include="object").columns

    # One-hot encoder
    encoder = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough",
    )

    # Fit on train only
    X_train = encoder.fit_transform(X_train)

    # Transform test
    X_test = encoder.transform(X_test)

    return X_train, X_test, y_train, y_test, encoder


# def preprocess_test_data(df: pd.DataFrame, encoder: ColumnTransformer) -> pd.DataFrame:

#     # drop the columns that are not useful for the model
#     df = df.drop(columns=COLUMNS_TO_DROP)

#     # Separate target
#     y_test = df[TARGET_COLUMN]
#     X_test = df.drop(columns=[TARGET_COLUMN])

#     # One-hot encode categorical features using the same encoder fitted on the training data
#     X_test = encoder.transform(X_test)

#     return X_test, y_test
