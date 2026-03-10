from prometheus_client import Counter, Histogram

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
