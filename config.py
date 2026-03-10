import os

# cleaning
COLUMNS_TO_DROP = ["loan_grade", "loan_int_rate", "id"]
TARGET_COLUMN = "loan_status"

# feature engineering
AGE_COLUMN = "person_age"
BINS_AGE = [18, 25, 35, 50, 125]
CATEGORIES_AGE = ["young", "adult", "mature", "senior"]
AGE_CATEGORY_COLUMN = "age_category"

CREDIT_DEFAULT_COLUMN = "cb_person_default_on_file"
CREDIT_DEFAULT_BINARY_COLUMN = "record_credit_default"

# Encoding
CATEGORICAL_DTYPE = "object"
OHE_HANDLE_UNKNOWN = "ignore"

CV_FOLDS = 5
MAX_EVALS = 10

RANDOM_STATE = 42
TEST_SIZE = 0.2

# MLflow — MLFLOW_TRACKING_URI env var overrides the default (useful in CI/CD)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = "Loan Prediction Approval Experiments"
MLFLOW_MODEL_NAME = "loan-approval-model"
F1_PROMOTION_THRESHOLD = 0.5  # minimum F1 to promote to champion
