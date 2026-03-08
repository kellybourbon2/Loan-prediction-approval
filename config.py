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
