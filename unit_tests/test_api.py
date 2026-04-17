import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

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


def _make_preprocessor():
    prep = MagicMock()
    prep.inference_transform.return_value = np.zeros((1, 9))
    prep.encoder.get_feature_names_out.return_value = [
        "cat__person_home_ownership_RENT",
        "cat__loan_intent_PERSONAL",
        "remainder__age_category",
        "remainder__person_income",
        "remainder__person_emp_length",
        "remainder__loan_amnt",
        "remainder__loan_percent_income",
        "remainder__cb_person_cred_hist_length",
        "remainder__record_credit_default",
    ]
    prep.encoder.transformers_ = [
        ("cat", MagicMock(), ["person_home_ownership", "loan_intent"])
    ]
    return prep


def _make_model(proba_default=0.3):
    """Mock model — predict_proba returns [[1-p, p]] where p = P(default)."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[1 - proba_default, proba_default]])
    model.feature_importances_ = np.ones(9) / 9
    return model


class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from api.app import app

        cls._patches = [
            patch("api.app.load_champion_model", return_value=_make_model()),
            patch("api.app.load_preprocessor_from_registry", return_value=_make_preprocessor()),
        ]
        for p in cls._patches:
            p.start()

        cls._ctx = TestClient(app)
        cls.client = cls._ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)
        for p in cls._patches:
            p.stop()

    # ── /health ──────────────────────────────────────────────────────
    def test_health_returns_ok(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"status": "ok"})

    # ── /predict ─────────────────────────────────────────────────────
    def test_predict_valid_payload(self):
        res = self.client.post("/predict", json=VALID_PAYLOAD)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("probability", data)
        self.assertIn("approved", data)
        self.assertIn("loan_status", data)
        self.assertIsInstance(data["probability"], float)
        self.assertGreaterEqual(data["probability"], 0.0)
        self.assertLessEqual(data["probability"], 1.0)

    def test_predict_approved_when_low_default_proba(self):
        """P(default)=0.1 → P(approval)=0.9 → approved=True."""
        from api.app import app
        app.state.model = _make_model(proba_default=0.1)
        res = self.client.post("/predict", json=VALID_PAYLOAD)
        app.state.model = _make_model()  # restore
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["approved"])

    def test_predict_rejected_when_high_default_proba(self):
        """P(default)=0.9 → P(approval)=0.1 → approved=False."""
        from api.app import app
        app.state.model = _make_model(proba_default=0.9)
        res = self.client.post("/predict", json=VALID_PAYLOAD)
        app.state.model = _make_model()  # restore
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json()["approved"])

    def test_predict_invalid_age(self):
        payload = {**VALID_PAYLOAD, "person_age": 10}  # below minimum (18)
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 422)

    def test_predict_invalid_ownership(self):
        payload = {**VALID_PAYLOAD, "person_home_ownership": "BOAT"}
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 422)

    def test_predict_missing_field(self):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "loan_amnt"}
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 422)

    # ── /predict/batch ───────────────────────────────────────────────
    def test_batch_returns_correct_structure(self):
        res = self.client.post("/predict/batch", json=[VALID_PAYLOAD, VALID_PAYLOAD])
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total"], 2)
        self.assertIn("approved_count", data)
        self.assertIn("rejected_count", data)
        self.assertIn("approval_rate", data)
        self.assertEqual(len(data["predictions"]), 2)

    def test_batch_counts_sum_to_total(self):
        res = self.client.post("/predict/batch", json=[VALID_PAYLOAD] * 5)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["approved_count"] + data["rejected_count"], data["total"])

    def test_batch_approval_rate_in_range(self):
        res = self.client.post("/predict/batch", json=[VALID_PAYLOAD] * 3)
        self.assertEqual(res.status_code, 200)
        rate = res.json()["approval_rate"]
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

    def test_batch_empty_list_raises_422(self):
        res = self.client.post("/predict/batch", json=[])
        self.assertEqual(res.status_code, 422)

    # ── /explain ─────────────────────────────────────────────────────
    def test_explain_returns_features(self):
        res = self.client.post("/explain", json=VALID_PAYLOAD)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("base_value", data)
        self.assertIn("features", data)
        self.assertGreater(len(data["features"]), 0)

    def test_explain_feature_structure(self):
        res = self.client.post("/explain", json=VALID_PAYLOAD)
        self.assertEqual(res.status_code, 200)
        for feat in res.json()["features"]:
            self.assertIn("feature", feat)
            self.assertIn("label", feat)
            self.assertIn("shap", feat)
            self.assertIsInstance(feat["shap"], float)

    def test_explain_base_value_in_range(self):
        res = self.client.post("/explain", json=VALID_PAYLOAD)
        self.assertEqual(res.status_code, 200)
        base = res.json()["base_value"]
        self.assertGreaterEqual(base, 0.0)
        self.assertLessEqual(base, 1.0)


if __name__ == "__main__":
    unittest.main()
