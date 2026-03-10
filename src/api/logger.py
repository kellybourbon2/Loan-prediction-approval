"""Structured prediction logger.

Writes one JSON line per prediction to logs/predictions.jsonl.
This file is the source of truth for drift analysis.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
        }
        if hasattr(record, "data"):
            entry.update(record.data)
        return json.dumps(entry)


def get_prediction_logger() -> logging.Logger:
    """Return (and lazily initialise) the prediction logger."""
    logger = logging.getLogger("predictions")
    if logger.handlers:
        return logger  # already configured

    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(log_dir / "predictions.jsonl", encoding="utf-8")
    handler.setFormatter(_JSONFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def log_prediction(
    inputs: dict,
    prediction: int,
    probability: float,
) -> None:
    """Log one prediction as a JSON line."""
    logger = get_prediction_logger()
    record = logging.LogRecord(
        name="predictions",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    )
    record.data = {
        "event": "prediction",
        **inputs,
        "loan_status": prediction,
        "probability": probability,
        "approved": bool(prediction),
    }
    logger.handle(record)
