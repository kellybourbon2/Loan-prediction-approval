"""Preprocessing tests to perform after fitting DataProcessor"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# define root dir so can see src functions
ROOT_DIR = Path(__file__).parent.parent  # noqa: E402
sys.path.insert(0, str(ROOT_DIR))

from src.data_processing.preprocessing import DataPreprocessor  # noqa: E402

from config import (  # noqa: E402
    COLUMNS_TO_DROP,
    TARGET_COLUMN,
    TEST_SIZE,
    CREDIT_DEFAULT_BINARY_COLUMN,
    CREDIT_DEFAULT_COLUMN,
    AGE_COLUMN,
    AGE_CATEGORY_COLUMN,
)


def generate_fake_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a fake dataset matching the real dataset schema — no S3 required."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "id": np.arange(n),
            AGE_COLUMN: rng.integers(18, 75, n),
            "person_income": rng.uniform(20000, 150000, n),
            "person_home_ownership": rng.choice(
                ["RENT", "OWN", "MORTGAGE", "OTHER"], n
            ),
            "person_emp_length": rng.uniform(0, 20, n),
            "loan_intent": rng.choice(
                [
                    "EDUCATION",
                    "MEDICAL",
                    "PERSONAL",
                    "VENTURE",
                    "DEBTCONSOLIDATION",
                    "HOMEIMPROVEMENT",
                ],
                n,
            ),
            "loan_grade": rng.choice(["A", "B", "C", "D", "E", "F", "G"], n),
            "loan_amnt": rng.uniform(500, 35000, n),
            "loan_int_rate": rng.uniform(5.0, 25.0, n),
            TARGET_COLUMN: rng.integers(0, 2, n),
            "loan_percent_income": rng.uniform(0.01, 0.5, n),
            CREDIT_DEFAULT_COLUMN: rng.choice(["Y", "N"], n),
            "cb_person_cred_hist_length": rng.integers(1, 30, n),
        }
    )


class TestDataPreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Generate fake df once for all tests — no S3 connection required."""
        cls.df = generate_fake_df(n=200)

    def setUp(self):
        """Initialize new DataPreprocessor before each test"""
        self.preprocessor = DataPreprocessor()

    # testing clean data
    def test_clean_data_drops_columns(self):
        """COLUMNS_TO_DROP should be removed"""
        result = self.preprocessor.clean_data(self.df.copy())
        for col in COLUMNS_TO_DROP:
            self.assertNotIn(col, result.columns)  # check each col individually

    def test_clean_data_removes_duplicates(self):
        """Duplicate rows should be removed"""
        df_with_dupes = pd.concat([self.df, self.df], ignore_index=True)
        result = self.preprocessor.clean_data(df_with_dupes)
        self.assertEqual(len(result), len(self.df))

    def test_clean_data_removes_nulls(self):
        """Rows with NaN should be dropped"""
        df = self.df.copy()
        df.loc[0, df.columns[0]] = None  # inject a null in first column
        result = self.preprocessor.clean_data(df)
        self.assertFalse(result.isnull().any().any())

    # testing feature_engineering
    def test_feature_engineering_creates_age_category(self):
        """AGE_CATEGORY_COLUMN  should be created"""
        df = self.preprocessor.clean_data(self.df.copy())
        result = self.preprocessor.feature_engineering(df)
        self.assertIn(AGE_CATEGORY_COLUMN, result.columns)

    def test_feature_engineering_drops_age(self):
        """Original 'age' column should be dropped"""
        df = self.preprocessor.clean_data(self.df.copy())
        result = self.preprocessor.feature_engineering(df)
        self.assertNotIn(AGE_COLUMN, result.columns)

    def test_feature_engineering_creates_credit_binary(self):
        """binary credit default column should be created"""
        df = self.preprocessor.clean_data(self.df.copy())
        result = self.preprocessor.feature_engineering(df)
        self.assertIn(CREDIT_DEFAULT_BINARY_COLUMN, result.columns)

    def test_feature_engineering_drops_credit(self):
        """Original CREDIT_DEFAULT_COLUMN should be dropped"""
        df = self.preprocessor.clean_data(self.df.copy())
        result = self.preprocessor.feature_engineering(df)
        self.assertNotIn(CREDIT_DEFAULT_COLUMN, result.columns)

    # testing split_data
    def test_split_data_correct_sizes(self):
        """Train/test split should respect TEST_SIZE from config"""
        df = self.preprocessor.clean_data(self.df.copy())
        df = self.preprocessor.feature_engineering(df)
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(df)

        total = len(df)
        expected_test = round(total * TEST_SIZE)
        expected_train = total - expected_test
        self.assertEqual(len(X_test), expected_test)  # dynamic, not hardcoded
        self.assertEqual(len(X_train), expected_train)

    def test_split_data_target_removed_from_X(self):
        """Target column should not appear in X_train or X_test"""
        df = self.preprocessor.clean_data(self.df.copy())
        df = self.preprocessor.feature_engineering(df)
        X_train, X_test, _, _ = self.preprocessor.split_data(df)
        self.assertNotIn(TARGET_COLUMN, X_train.columns)
        self.assertNotIn(TARGET_COLUMN, X_test.columns)

    def test_split_data_y_length(self):
        """y_train and y_test should match X length"""
        df = self.preprocessor.clean_data(self.df.copy())
        df = self.preprocessor.feature_engineering(df)
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(df)
        self.assertEqual(len(y_train), len(X_train))
        self.assertEqual(len(y_test), len(X_test))

    # testing normalize_data on a random variable
    # ...
