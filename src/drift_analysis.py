"""Drift analysis on prediction logs.

Reads logs/predictions.jsonl and compares the distribution of recent predictions
against a reference window to detect feature drift and prediction drift.

Usage:
    uv run python src/drift_analysis.py
    uv run python src/drift_analysis.py --log-file logs/predictions.jsonl --window 500
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Features to monitor for drift
NUMERICAL_FEATURES = [
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "probability",
]
CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file",
]
DRIFT_ALPHA = 0.05  # p-value threshold for KS test
PSI_THRESHOLD = 0.2  # PSI > 0.2 = significant drift


def load_logs(log_file: Path) -> pd.DataFrame:
    records = []
    with open(log_file, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    if not records:
        raise ValueError(f"No valid records found in {log_file}")
    return pd.DataFrame(records)


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index — measures distributional shift.

    PSI < 0.1  : no drift
    PSI < 0.2  : minor drift
    PSI >= 0.2 : significant drift
    """
    ref_counts, bin_edges = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = (ref_counts + 1e-6) / len(reference)
    cur_pct = (cur_counts + 1e-6) / len(current)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def analyse_numerical_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    results = {}
    for col in NUMERICAL_FEATURES:
        if col not in reference.columns or col not in current.columns:
            continue
        ref_vals = reference[col].dropna().values
        cur_vals = current[col].dropna().values

        ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
        psi = compute_psi(ref_vals, cur_vals)

        results[col] = {
            "ks_statistic": round(ks_stat, 4),
            "p_value": round(p_value, 4),
            "psi": round(psi, 4),
            "ref_mean": round(float(np.mean(ref_vals)), 4),
            "cur_mean": round(float(np.mean(cur_vals)), 4),
            "drift_detected": p_value < DRIFT_ALPHA or psi >= PSI_THRESHOLD,
        }
    return results


def analyse_categorical_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    results = {}
    for col in CATEGORICAL_FEATURES:
        if col not in reference.columns or col not in current.columns:
            continue
        ref_dist = reference[col].value_counts(normalize=True).to_dict()
        cur_dist = current[col].value_counts(normalize=True).to_dict()

        all_cats = set(ref_dist) | set(cur_dist)
        max_shift = max(abs(cur_dist.get(c, 0) - ref_dist.get(c, 0)) for c in all_cats)
        results[col] = {
            "reference_distribution": {k: round(v, 4) for k, v in ref_dist.items()},
            "current_distribution": {k: round(v, 4) for k, v in cur_dist.items()},
            "max_category_shift": round(max_shift, 4),
            "drift_detected": max_shift > 0.1,
        }
    return results


def analyse_prediction_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    ref_approval = (
        reference["approved"].mean() if "approved" in reference.columns else None
    )
    cur_approval = current["approved"].mean() if "approved" in current.columns else None
    if ref_approval is None or cur_approval is None:
        return {}
    return {
        "ref_approval_rate": round(ref_approval, 4),
        "cur_approval_rate": round(cur_approval, 4),
        "shift": round(cur_approval - ref_approval, 4),
        "drift_detected": abs(cur_approval - ref_approval) > 0.05,
    }


def print_report(numerical: dict, categorical: dict, prediction: dict) -> None:
    print("\n" + "=" * 60)
    print("DRIFT ANALYSIS REPORT")
    print("=" * 60)

    print("\n── Prediction Drift ──")
    if prediction:
        flag = "🔴 DRIFT" if prediction["drift_detected"] else "✅ OK"
        print(
            f"  Approval rate: {prediction['ref_approval_rate']:.1%} → {prediction['cur_approval_rate']:.1%}  {flag}"
        )
    else:
        print("  No data.")

    print("\n── Numerical Feature Drift (KS test + PSI) ──")
    for col, r in numerical.items():
        flag = "🔴 DRIFT" if r["drift_detected"] else "✅ OK"
        print(
            f"  {col:<30} PSI={r['psi']:.3f}  p={r['p_value']:.3f}  mean {r['ref_mean']:.2f}→{r['cur_mean']:.2f}  {flag}"
        )

    print("\n── Categorical Feature Drift ──")
    for col, r in categorical.items():
        flag = "🔴 DRIFT" if r["drift_detected"] else "✅ OK"
        print(f"  {col:<30} max_shift={r['max_category_shift']:.3f}  {flag}")

    drifted = (
        sum(r["drift_detected"] for r in numerical.values())
        + sum(r["drift_detected"] for r in categorical.values())
        + (1 if prediction.get("drift_detected") else 0)
    )
    print(
        f"\n{'🔴 DRIFT DETECTED on ' + str(drifted) + ' feature(s)!' if drifted else '✅ No significant drift detected.'}"
    )
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Drift analysis on prediction logs")
    parser.add_argument("--log-file", type=Path, default=Path("logs/predictions.jsonl"))
    parser.add_argument(
        "--window", type=int, default=500, help="Size of each comparison window"
    )
    args = parser.parse_args()

    if not args.log_file.exists():
        print(f"Log file not found: {args.log_file}")
        sys.exit(1)

    df = load_logs(args.log_file)
    print(f"Loaded {len(df)} predictions from {args.log_file}")

    if len(df) < args.window * 2:
        print(
            f"Not enough data for drift analysis (need {args.window * 2} predictions, got {len(df)})"
        )
        sys.exit(0)

    reference = df.iloc[: args.window]
    current = df.iloc[-args.window :]

    numerical = analyse_numerical_drift(reference, current)
    categorical = analyse_categorical_drift(reference, current)
    prediction = analyse_prediction_drift(reference, current)

    print_report(numerical, categorical, prediction)


if __name__ == "__main__":
    main()
