import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# define root dir so can see src module
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import (  # noqa: E402
    AGE_COLUMN,
    AGE_CATEGORY_COLUMN,
    BINS_AGE,
    CATEGORICAL_DTYPE,
    CATEGORIES_AGE,
    CREDIT_DEFAULT_COLUMN,
    CREDIT_DEFAULT_BINARY_COLUMN,
    COLUMNS_TO_DROP,
    OHE_HANDLE_UNKNOWN,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = None
        self.ordinal_encoder = None
        self.numerical_cols = None

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicates, missing values and useless columns"""
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        df.drop(columns=COLUMNS_TO_DROP, inplace=True)
        return df

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones"""

        # age into categories
        df[AGE_CATEGORY_COLUMN] = pd.cut(
            df[AGE_COLUMN], bins=BINS_AGE, labels=CATEGORIES_AGE
        ).astype(str)
        df.drop(columns=[AGE_COLUMN], inplace=True)  # drop old age column

        # credit default into binary variable
        df[CREDIT_DEFAULT_BINARY_COLUMN] = (df[CREDIT_DEFAULT_COLUMN] == "Y").astype(
            int
        )
        df.drop(columns=[CREDIT_DEFAULT_COLUMN], inplace=True)  # drop old column

        return df

    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ):
        """Split the data into training and testing sets"""
        y = df[TARGET_COLUMN]
        X = df.drop(columns=[TARGET_COLUMN])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test

    def normalize_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """Normalize only continuous numerical features"""
        self.numerical_cols = [
            col
            for col in X_train.select_dtypes(include=["int64", "float64"]).columns
            if col != CREDIT_DEFAULT_BINARY_COLUMN
        ]  # exclude binary column created

        X_train[self.numerical_cols] = self.scaler.fit_transform(
            X_train[self.numerical_cols]
        )
        X_test[self.numerical_cols] = self.scaler.transform(X_test[self.numerical_cols])

        return X_train, X_test

    def inference_transform(self, df: pd.DataFrame):
        """Apply feature engineering + normalization + encoding for a single inference sample.

        Skips clean_data (no columns to drop at inference time).
        Requires the preprocessor to be already fitted (after preprocessing_pipeline).
        """
        df = df.copy()
        df = self.feature_engineering(df)
        df[self.numerical_cols] = self.scaler.transform(df[self.numerical_cols])
        df[AGE_CATEGORY_COLUMN] = self.ordinal_encoder.transform(
            df[[AGE_CATEGORY_COLUMN]]
        )
        return self.encoder.transform(df)

    def feature_encoding(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """Transform categorical columns so they can be processed by the model"""

        # Ordinal encoding of 'AGE_CATEGORY_COLUMN', fit on train, transform both
        self.ordinal_encoder = OrdinalEncoder(categories=[CATEGORIES_AGE])
        X_train[AGE_CATEGORY_COLUMN] = self.ordinal_encoder.fit_transform(
            X_train[[AGE_CATEGORY_COLUMN]]
        )
        X_test[AGE_CATEGORY_COLUMN] = self.ordinal_encoder.transform(
            X_test[[AGE_CATEGORY_COLUMN]]
        )

        # One-hot encoding of remaining categorical columns
        categorical_cols = [
            col
            for col in X_train.select_dtypes(include=[CATEGORICAL_DTYPE]).columns
            if col != AGE_CATEGORY_COLUMN
        ]
        self.encoder = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown=OHE_HANDLE_UNKNOWN),
                    categorical_cols,
                )
            ],
            remainder="passthrough",
        )

        # Fit on train, and reuse to transform test
        X_train_encoded = self.encoder.fit_transform(X_train)
        X_test_encoded = self.encoder.transform(X_test)

        return X_train_encoded, X_test_encoded

    def preprocessing_pipeline(self, df: pd.DataFrame):
        """Apply the full preprocessing pipeline:
        clean -> feature engineering --> normalize --> encode"""
        df = self.clean_data(df)
        df = self.feature_engineering(df)
        X_train, X_test, y_train, y_test = self.split_data(df)
        X_train, X_test = self.normalize_data(X_train, X_test)
        X_train, X_test = self.feature_encoding(X_train, X_test)
        return X_train, X_test, y_train, y_test, self


def preprocess_data(df: pd.DataFrame):
    """Wrapper function to apply the full preprocessing pipeline.

    Returns X_train, X_test, y_train, y_test, fitted_preprocessor.
    """
    preprocessor = DataPreprocessor()
    return preprocessor.preprocessing_pipeline(df)


if __name__ == "__main__":
    from data_load import DataLoader

    loader = DataLoader()
    df = loader.load_train()

    # preprocessing
    datapreprocessor = DataPreprocessor()
    df_processed = datapreprocessor.preprocessing_pipeline(df)
    print(df_processed)
