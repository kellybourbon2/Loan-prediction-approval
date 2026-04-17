from prometheus_client import Counter, Gauge, Histogram

# Count predictions by outcome (approved / rejected)
PREDICTION_COUNTER = Counter(
    "loan_predictions_total",
    "Number of loan predictions made",
    ["result"],  # label: "approved" or "rejected"
)

# Distribution of approval probabilities — detects model drift over time
PROBABILITY_HISTOGRAM = Histogram(
    "loan_prediction_probability",
    "Distribution of loan approval probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Count prediction errors
PREDICTION_ERRORS = Counter(
    "loan_prediction_errors_total",
    "Number of errors during prediction",
)

# Running approval rate (updated after each prediction)
APPROVAL_RATE_GAUGE = Gauge(
    "loan_approval_rate",
    "Rolling approval rate over last 100 predictions",
)

# Feature distribution histograms for data drift detection
INCOME_HISTOGRAM = Histogram(
    "loan_request_income",
    "Distribution of applicant annual income",
    buckets=[10000, 25000, 40000, 60000, 80000, 100000, 150000, 200000],
)

LOAN_AMOUNT_HISTOGRAM = Histogram(
    "loan_request_amount",
    "Distribution of requested loan amounts",
    buckets=[1000, 5000, 10000, 20000, 35000, 50000, 75000, 100000],
)

LTI_HISTOGRAM = Histogram(
    "loan_request_lti_ratio",
    "Distribution of loan-to-income ratios",
    buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
)

# Batch endpoint metrics
BATCH_SIZE_HISTOGRAM = Histogram(
    "loan_batch_size",
    "Number of applications per batch request",
    buckets=[1, 5, 10, 25, 50, 100, 500],
)
