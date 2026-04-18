"""
Integration tests — spin up uvicorn in a subprocess and hit the real HTTP stack.
Requires a running MLflow registry with a @champion model, or the SKIP_INTEGRATION
env var set to '1' to skip automatically (e.g. in environments without MLflow).
"""

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

import requests

BASE_URL = os.getenv("INTEGRATION_API_URL", "http://127.0.0.1:8765")

VALID_PAYLOAD = {
    "person_age": 30,
    "person_income": 60000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5.0,
    "loan_intent": "PERSONAL",
    "loan_amnt": 10000,
    "loan_percent_income": 0.17,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 4,
}


def _wait_for_api(url: str, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    return False


@unittest.skipIf(os.getenv("SKIP_INTEGRATION") == "1", "Integration tests skipped")
class TestIntegration(unittest.TestCase):
    _server: subprocess.Popen | None = None

    @classmethod
    def setUpClass(cls):
        # If a URL is provided externally (e.g. a deployed env), skip launching
        if os.getenv("INTEGRATION_API_URL"):
            if not _wait_for_api(BASE_URL, timeout=10):
                raise RuntimeError(f"External API at {BASE_URL} not reachable")
            return

        root = Path(__file__).parent.parent
        cls._server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "src.api.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8765",
            ],
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if not _wait_for_api(BASE_URL, timeout=60):
            cls._server.terminate()
            raise RuntimeError("API did not start in time")

    @classmethod
    def tearDownClass(cls):
        if cls._server:
            cls._server.terminate()
            cls._server.wait()

    # ── /health ──────────────────────────────────────────────────────────
    def test_health_ok(self):
        r = requests.get(f"{BASE_URL}/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    # ── /predict ─────────────────────────────────────────────────────────
    def test_predict_returns_valid_response(self):
        r = requests.post(f"{BASE_URL}/predict", json=VALID_PAYLOAD)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("approved", data)
        self.assertIn("probability", data)
        self.assertIn("loan_status", data)
        self.assertIsInstance(data["probability"], float)
        self.assertGreaterEqual(data["probability"], 0.0)
        self.assertLessEqual(data["probability"], 1.0)

    def test_predict_invalid_payload_returns_422(self):
        r = requests.post(f"{BASE_URL}/predict", json={"bad": "payload"})
        self.assertEqual(r.status_code, 422)

    def test_predict_error_does_not_leak_traceback(self):
        """500 responses must not expose internal Python tracebacks."""
        # Send a valid-schema but numerically extreme payload to stress the model
        payload = {**VALID_PAYLOAD, "person_income": -1}
        r = requests.post(f"{BASE_URL}/predict", json=payload)
        if r.status_code == 500:
            detail = r.json().get("detail", "")
            self.assertNotIn("Traceback", detail)
            self.assertNotIn("File ", detail)

    # ── /predict/batch ───────────────────────────────────────────────────
    def test_batch_predict_two_items(self):
        r = requests.post(
            f"{BASE_URL}/predict/batch", json=[VALID_PAYLOAD, VALID_PAYLOAD]
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["total"], 2)
        self.assertEqual(data["approved_count"] + data["rejected_count"], 2)
        self.assertGreaterEqual(data["approval_rate"], 0.0)
        self.assertLessEqual(data["approval_rate"], 1.0)

    def test_batch_empty_returns_422(self):
        r = requests.post(f"{BASE_URL}/predict/batch", json=[])
        self.assertEqual(r.status_code, 422)

    # ── /explain ─────────────────────────────────────────────────────────
    def test_explain_returns_features(self):
        r = requests.post(f"{BASE_URL}/explain", json=VALID_PAYLOAD)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("base_value", data)
        self.assertGreater(len(data["features"]), 0)
        for feat in data["features"]:
            self.assertIn("feature", feat)
            self.assertIn("shap", feat)

    # ── /metrics ─────────────────────────────────────────────────────────
    def test_metrics_endpoint_reachable(self):
        r = requests.get(f"{BASE_URL}/metrics")
        self.assertEqual(r.status_code, 200)
        self.assertIn("loan_predictions_total", r.text)


if __name__ == "__main__":
    unittest.main()
