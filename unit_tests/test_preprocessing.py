import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# define root dir so can see src functions
ROOT_DIR = Path(__file__).parent.parent  # noqa: E402
sys.path.insert(0, str(ROOT_DIR))

from src.data_processing.data_load import DataLoader  # noqa: E402
from src.data_processing.preprocessing import DataPreprocessor  # noqa: E402

from config import (  # noqa: E402
    COLUMNS_TO_DROP,
    TARGET_COLUMN,
    TEST_SIZE,
    CREDIT_DEFAULT_BINARY_COLUMN,
    CREDIT_DEFAULT_COLUMN,
    AGE_COLUMN,
    AGE_CATEGORY_COLUMN,
)  # noqa: E402


# function to generate a fake df similar than original one
def generate_like(df_real: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    """Generate a fake df with same columns, dtypes and value range as the original"""
    fake_data = {}
    for col in df_real.columns:
        if df_real[col].dtype == "object":
            fake_data[col] = np.random.choice(df_real[col].dropna().unique(), n)
        elif df_real[col].dtype in ["int64", "int32"]:
            fake_data[col] = np.random.randint(
                df_real[col].min(), df_real[col].max() + 1, n
            )
        elif df_real[col].dtype in ["float64", "float32"]:
            fake_data[col] = np.random.uniform(
                df_real[col].min(), df_real[col].max(), n
            )
    return pd.DataFrame(fake_data)


class TestDataPreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load real df once and generate similar fake df for all tests"""
        data_load = DataLoader()
        df_real = data_load.load_train()
        cls.df = generate_like(df_real, n=200)

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
