"""
Monitoring metrics runner (CI entrypoint).

Loads the healthy baseline and a window of new data, runs the MetricComputer metrics,
compares every metric against its alert threshold, writes the results to a timestamped
JSON file, and exits non-zero when any CRITICAL alert fires so the GitHub Actions
workflow can open an alert issue.

Usage:
    python3 scripts/compute_metrics.py
    python3 scripts/compute_metrics.py --new-data data/demand_enriched_week4.parquet \
        --start 2026-02-02 --end 2026-02-28

Environment overrides (used by CI): NEW_DATA, WINDOW_START, WINDOW_END.
"""

import argparse
import json
import os
import sys

import pandas as pd

# Allow running both as `python3 scripts/compute_metrics.py` and from inside scripts/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric_template import (  # noqa: E402
    MetricComputer,
    PRED_PROXY_COL,
    LAG_NULL_COLUMNS,
    CRITICAL_NULL_COLUMNS,
)

# --- Paths and the drift window we monitor by default (Feb 2-28, 2026) ---
BASELINE_PATH = "data/demand_enriched_baseline.parquet"
DEFAULT_NEW_DATA = "data/demand_enriched_week4.parquet"
DEFAULT_START = "2026-02-02"
DEFAULT_END = "2026-02-28"

# Distribution-drift scan covers the target plus the day-scale engineered features,
# which is where the injected drift is strongest (roll_mean_1day / lag_1day).
DRIFT_SCAN_FEATURES = ["trip_count", "roll_mean_1day", "lag_1day", "lag_1week"]

# --- Alert thresholds (grounded in BASELINE_METRICS.md and the measured baseline) ---
ACC_DROP_WARN = 0.05          # overall accuracy drop vs baseline
ACC_DROP_CRIT = 0.10
ZONE_DROP = 0.10              # a zone "degraded" if its accuracy fell this much vs baseline
ZONE_DROP_WARN_COUNT = 1      # >=1 degraded zone warns
ZONE_DROP_CRIT_COUNT = 5      # >=5 degraded zones is critical
NULL_WARN, NULL_CRIT = 0.005, 0.01
LAG_NULL_WARN, LAG_NULL_CRIT = 0.01, 0.02
KS_STAT_ACTIONABLE = 0.10    # gate KS significance on effect size to avoid large-N false alarms
PSI_WARN, PSI_CRIT = 0.10, 0.25
FRESHNESS_WARN_MIN, FRESHNESS_CRIT_MIN = 30, 60
DUP_WARN, DUP_CRIT = 0.0, 0.005


def load_window(path, start, end):
    """Load a parquet file and slice it to [start, end] on time_bucket when given."""
    df = pd.read_parquet(path)
    if start and end:
        mask = (df["time_bucket"] >= start) & (df["time_bucket"] <= f"{end} 23:59:59")
        df = df[mask].copy()
    return df


def predictions_and_actuals(df):
    """Derive the proxy prediction (frozen zone_slot_baseline) and the actual counts."""
    return df[PRED_PROXY_COL].to_numpy(), df["trip_count"].to_numpy()


def evaluate_alerts(baseline_metrics, new_metrics, drift_scan):
    """Compare new metrics to baseline/thresholds. Returns a list of alert dicts."""
    alerts = []

    def add(level, metric, detail):
        alerts.append({"level": level, "metric": metric, "detail": detail})

    # 1. Overall accuracy vs baseline
    base_acc = baseline_metrics.get("accuracy")
    new_acc = new_metrics.get("accuracy")
    if base_acc is not None and new_acc is not None:
        drop = base_acc - new_acc
        if drop > ACC_DROP_CRIT:
            add("CRITICAL", "accuracy", f"dropped {drop:.3f} (base {base_acc:.3f} -> {new_acc:.3f})")
        elif drop > ACC_DROP_WARN:
            add("WARNING", "accuracy", f"dropped {drop:.3f} (base {base_acc:.3f} -> {new_acc:.3f})")

    # 2. Accuracy by zone — count zones that fell off their own baseline
    base_zone = baseline_metrics.get("accuracy_by_zone", {})
    new_zone = new_metrics.get("accuracy_by_zone", {})
    degraded = sorted(
        z for z, a in new_zone.items()
        if z in base_zone and (base_zone[z] - a) > ZONE_DROP
    )
    if len(degraded) >= ZONE_DROP_CRIT_COUNT:
        add("CRITICAL", "accuracy_by_zone", f"{len(degraded)} zones dropped >{ZONE_DROP:.0%}: {degraded}")
    elif len(degraded) >= ZONE_DROP_WARN_COUNT:
        add("WARNING", "accuracy_by_zone", f"{len(degraded)} zones dropped >{ZONE_DROP:.0%}: {degraded}")

    # 3. Null rates on critical / lag columns
    for col, rate in new_metrics.get("null_rates", {}).items():
        warn, crit = (LAG_NULL_WARN, LAG_NULL_CRIT) if col in LAG_NULL_COLUMNS else (NULL_WARN, NULL_CRIT)
        if rate > crit:
            add("CRITICAL", "null_rate", f"{col} null rate {rate:.3%}")
        elif rate > warn:
            add("WARNING", "null_rate", f"{col} null rate {rate:.3%}")

    # 4/5. Distribution drift (KS gated by effect size, plus PSI) across scanned features
    for feat, res in drift_scan.items():
        psi = res["psi"]
        if psi > PSI_CRIT:
            add("CRITICAL", "psi", f"{feat} PSI={psi:.3f}")
        elif psi > PSI_WARN:
            add("WARNING", "psi", f"{feat} PSI={psi:.3f}")
        ks = res["ks"]
        if ks["significant"] and ks["statistic"] > KS_STAT_ACTIONABLE:
            add("WARNING", "ks_test", f"{feat} KS stat={ks['statistic']:.3f} p={ks['p_value']:.1e}")

    # 6. Prediction distribution collapse
    pred = new_metrics.get("prediction_distribution", {})
    if pred.get("collapsed"):
        add("CRITICAL", "prediction_distribution", f"collapsed: std={pred['std']:.2f} vs baseline {pred['baseline_std']:.2f}")

    # 7. Data freshness
    age = new_metrics.get("data_freshness", {}).get("age_minutes", 0.0)
    if age > FRESHNESS_CRIT_MIN:
        add("CRITICAL", "data_freshness", f"most recent record is {age:.0f} min old")
    elif age > FRESHNESS_WARN_MIN:
        add("WARNING", "data_freshness", f"most recent record is {age:.0f} min old")

    # 8. Duplicate rate
    dup = new_metrics.get("duplicate_rate", {}).get("rate", 0.0)
    if dup > DUP_CRIT:
        add("CRITICAL", "duplicate_rate", f"duplicate rate {dup:.3%}")
    elif dup > DUP_WARN:
        add("WARNING", "duplicate_rate", f"duplicate rate {dup:.3%}")

    return alerts


def main():
    parser = argparse.ArgumentParser(description="Run monitoring metrics and check thresholds.")
    parser.add_argument("--baseline", default=BASELINE_PATH)
    parser.add_argument("--new-data", default=os.getenv("NEW_DATA", DEFAULT_NEW_DATA))
    parser.add_argument("--start", default=os.getenv("WINDOW_START", DEFAULT_START))
    parser.add_argument("--end", default=os.getenv("WINDOW_END", DEFAULT_END))
    args = parser.parse_args()

    print("=" * 70)
    print("MONITORING METRICS")
    print("=" * 70)
    print(f"baseline : {args.baseline}")
    print(f"new data : {args.new_data}  window [{args.start} .. {args.end}]")

    baseline_df = load_window(args.baseline, None, None)
    new_df = load_window(args.new_data, args.start, args.end)
    print(f"baseline rows={len(baseline_df)} | new rows={len(new_df)}")

    mc = MetricComputer(baseline_df)
    b_pred, b_act = predictions_and_actuals(baseline_df)
    n_pred, n_act = predictions_and_actuals(new_df)

    baseline_metrics = mc.compute_all_metrics(baseline_df, predictions=b_pred, actuals=b_act)
    new_metrics = mc.compute_all_metrics(new_df, predictions=n_pred, actuals=n_act)

    # Multi-feature drift scan (target + day-scale engineered features)
    drift_scan = {
        feat: {"psi": mc.metric_5_psi(new_df, feature=feat),
               "ks": mc.metric_4_ks_test(new_df, feature=feat)}
        for feat in DRIFT_SCAN_FEATURES if feat in new_df.columns
    }

    alerts = evaluate_alerts(baseline_metrics, new_metrics, drift_scan)

    # --- Console summary ---
    print("\n--- Headline metrics (new window) ---")
    print(f"accuracy            : {new_metrics['accuracy']:.3f}  (baseline {baseline_metrics['accuracy']:.3f})")
    print(f"accuracy_by_zone    : {len(new_metrics['accuracy_by_zone'])} zones")
    print(f"psi(trip_count)     : {drift_scan.get('trip_count', {}).get('psi', float('nan')):.3f}")
    print(f"psi(roll_mean_1day) : {drift_scan.get('roll_mean_1day', {}).get('psi', float('nan')):.3f}")
    print(f"duplicate_rate      : {new_metrics['duplicate_rate']['rate']:.3%}")
    print(f"freshness (min)     : {new_metrics['data_freshness']['age_minutes']:.0f}")

    print("\n--- Alerts ---")
    if not alerts:
        print("none — system healthy")
    for a in alerts:
        print(f"[{a['level']}] {a['metric']}: {a['detail']}")

    # --- Persist results ---
    n_crit = sum(1 for a in alerts if a["level"] == "CRITICAL")
    n_warn = sum(1 for a in alerts if a["level"] == "WARNING")
    payload = {
        "window": {"start": args.start, "end": args.end, "new_data": args.new_data},
        "baseline_metrics": baseline_metrics,
        "new_metrics": new_metrics,
        "drift_scan": drift_scan,
        "alerts": alerts,
        "summary": {"critical": n_crit, "warning": n_warn},
    }
    out_path = f"metrics-{args.start}_to_{args.end}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nwrote {out_path}  (critical={n_crit}, warning={n_warn})")

    # Non-zero exit on any CRITICAL alert -> CI opens an issue.
    if n_crit > 0:
        print("CRITICAL alerts present — failing the job to trigger ops alert.")
        sys.exit(1)


if __name__ == "__main__":
    main()
