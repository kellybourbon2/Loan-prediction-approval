"""Structured prediction logger.

Writes one JSON line per prediction to logs/predictions.jsonl.
This file is the source of truth for drift analysis.
Syncs to S3 every 10 predictions and on a 60-second periodic timer
(background threads, non-blocking) — limits data loss on pod restart.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

_prediction_count = 0
_SYNC_EVERY = 10


def _sync_to_s3(log_path: Path) -> None:
    """Upload prediction logs to S3 — runs in background, never raises."""
    try:
        import boto3

        bucket = os.getenv("AWS_BUCKET_NAME")
        endpoint = os.getenv("AWS_S3_ENDPOINT")
        if not bucket or not endpoint:
            return
        s3 = boto3.client("s3", endpoint_url=f"https://{endpoint}")
        s3.upload_file(str(log_path), bucket, "logs/predictions.jsonl")
    except Exception:
        pass


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
        }
        if hasattr(record, "data"):
            entry.update(record.data)
        return json.dumps(entry)


def _start_periodic_sync(log_path: Path, interval: int = 60) -> None:
    """Spawn a daemon thread that syncs to S3 every `interval` seconds."""

    def _loop():
        import time

        while True:
            time.sleep(interval)
            _sync_to_s3(log_path)

    threading.Thread(target=_loop, daemon=True).start()


def get_prediction_logger() -> logging.Logger:
    """Return (and lazily initialise) the prediction logger."""
    logger = logging.getLogger("predictions")
    if logger.handlers:
        return logger  # already configured

    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "predictions.jsonl"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(_JSONFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    _start_periodic_sync(log_path)
    return logger


def log_prediction(
    inputs: dict,
    prediction: int,
    probability: float,
) -> None:
    """Log one prediction as a JSON line, sync to S3 every 100 predictions."""
    global _prediction_count

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

    _prediction_count += 1
    if _prediction_count % _SYNC_EVERY == 0:
        log_path = Path(os.getenv("LOG_DIR", "logs")) / "predictions.jsonl"
        threading.Thread(target=_sync_to_s3, args=(log_path,), daemon=True).start()
