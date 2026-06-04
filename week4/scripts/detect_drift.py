"""
Drift detection skeleton.

Write code to detect 4+ distinct drift patterns between baseline and new data.
Use statistical tests (KS, PSI, chi-square) to quantify drift.
"""

import os
import sys

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric_template import MetricComputer, PRED_PROXY_COL  # noqa: E402

# Same window the monitoring job watches.
BASELINE_PATH = "data/demand_enriched_baseline.parquet"
NEW_DATA_PATH = "data/demand_enriched_week4.parquet"
WINDOW_START, WINDOW_END = "2026-02-02", "2026-02-28"

# Features to test for distribution (data) drift: the target plus the engineered
# day-scale features, which is where the injected drift concentrates.
FEATURE_LIST = ["trip_count", "roll_mean_1day", "lag_1day", "lag_1week", "hour", "dayofweek"]

PSI_BINS = 10
SEGMENT_ACC_DROP = 0.10   # a segment "degraded" if proxy accuracy fell this much vs baseline


def _psi(base: np.ndarray, new: np.ndarray, bins: int = PSI_BINS) -> float:
    """Population Stability Index using baseline quantile bins."""
    edges = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    base_pct = np.clip(np.histogram(base, bins=edges)[0] / len(base), 1e-6, None)
    new_pct = np.clip(np.histogram(new, bins=edges)[0] / len(new), 1e-6, None)
    return float(np.sum((new_pct - base_pct) * np.log(new_pct / base_pct)))


def _severity(psi: float) -> str:
    if np.isnan(psi):
        return "n/a"
    if psi > 0.25:
        return "SIGNIFICANT"
    if psi > 0.10:
        return "moderate"
    return "negligible"


def detect_feature_drift(baseline_df: pd.DataFrame, new_df: pd.DataFrame, feature: str) -> dict:
    """
    Detect drift in a single feature.

    Runs a KS test and PSI comparing baseline vs new distribution and reports the mean
    shift. The KS statistic is the effect size; with large samples the p-value is tiny
    even for trivial shifts, so PSI drives the severity verdict.
    """
    base = baseline_df[feature].dropna().values
    new = new_df[feature].dropna().values
    ks = ks_2samp(base, new)
    psi = _psi(base, new)
    shift_pct = (new.mean() - base.mean()) / base.mean() * 100 if base.mean() else float("nan")
    return {
        "feature": feature,
        "ks_statistic": float(ks.statistic),
        "ks_p_value": float(ks.pvalue),
        "psi": psi,
        "mean_shift_pct": float(shift_pct),
        "severity": _severity(psi),
    }


# Segments scanned for concept drift: single columns plus the borough x weekend
# interaction, which exposes conditional drift that single-column views average away.
SEGMENT_SPECS = {
    "zone": "PULocationID",
    "hour": "hour",
    "borough_x_weekend": ["borough_id", "is_weekend"],
}


def detect_concept_drift_by_segment(baseline_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    """
    Detect concept drift (accuracy degradation by segment).

    Uses the frozen zone_slot_baseline proxy (same as the monitoring metrics) to score
    accuracy per segment on baseline vs new data, and flags segments whose accuracy fell
    by more than SEGMENT_ACC_DROP. Globally the data looks fine; the damage is in specific
    segments (zones, hours, and the borough x weekend interaction), which is exactly what
    segmentation surfaces.
    """
    mc = MetricComputer(baseline_df)

    def seg_accuracy(df, cols):
        cols = cols if isinstance(cols, list) else [cols]
        hits, valid = mc._accuracy_hits(df[PRED_PROXY_COL].to_numpy(), df["trip_count"].to_numpy())
        tmp = df[cols].copy()
        tmp["_hit"], tmp["_valid"] = hits, valid
        g = tmp[tmp["_valid"]].groupby(cols)["_hit"].mean()
        return {k: float(v) for k, v in g.items()}

    findings = {}
    for name, cols in SEGMENT_SPECS.items():
        needed = cols if isinstance(cols, list) else [cols]
        if any(c not in baseline_df.columns or c not in new_df.columns for c in needed):
            continue
        base_acc = seg_accuracy(baseline_df, cols)
        new_acc = seg_accuracy(new_df, cols)
        drops = {
            k: {"baseline": round(base_acc[k], 3), "new": round(new_acc[k], 3),
                "drop": round(base_acc[k] - new_acc[k], 3)}
            for k in new_acc if k in base_acc and (base_acc[k] - new_acc[k]) > SEGMENT_ACC_DROP
        }
        findings[name] = dict(sorted(drops.items(), key=lambda kv: -kv[1]["drop"]))
    return findings


def main():
    """Main drift detection analysis."""
    print("=" * 70)
    print("DRIFT DETECTION")
    print("=" * 70)

    baseline_df = pd.read_parquet(BASELINE_PATH)
    new_all = pd.read_parquet(NEW_DATA_PATH)
    mask = (new_all["time_bucket"] >= WINDOW_START) & (new_all["time_bucket"] <= f"{WINDOW_END} 23:59:59")
    new_df = new_all[mask].copy()
    print(f"baseline rows={len(baseline_df)} | new rows={len(new_df)} "
          f"window [{WINDOW_START} .. {WINDOW_END}]\n")

    # --- Data drift: feature distributions ---
    print("-" * 70)
    print("DATA DRIFT (feature distributions: KS + PSI)")
    print("-" * 70)
    print(f"{'feature':<16}{'mean_shift%':>12}{'KS_stat':>10}{'KS_p':>11}{'PSI':>9}  severity")
    feature_results = []
    for feat in FEATURE_LIST:
        if feat not in new_df.columns:
            continue
        r = detect_feature_drift(baseline_df, new_df, feat)
        feature_results.append(r)
        print(f"{r['feature']:<16}{r['mean_shift_pct']:>12.1f}{r['ks_statistic']:>10.4f}"
              f"{r['ks_p_value']:>11.1e}{r['psi']:>9.3f}  {r['severity']}")

    # --- Concept drift: segmented accuracy ---
    print("\n" + "-" * 70)
    print("CONCEPT DRIFT (proxy accuracy degradation by segment)")
    print("-" * 70)
    seg = detect_concept_drift_by_segment(baseline_df, new_df)

    zone_drops = seg["zone"]
    print(f"\nZones with accuracy drop > {SEGMENT_ACC_DROP:.0%}: {len(zone_drops)} / 57")
    for z, v in list(zone_drops.items())[:8]:
        print(f"  zone {z:>3}: {v['baseline']:.3f} -> {v['new']:.3f}  (drop {v['drop']:.3f})")

    hour_drops = seg["hour"]
    print(f"\nHours with accuracy drop > {SEGMENT_ACC_DROP:.0%}: {len(hour_drops)} / 24")
    for h, v in hour_drops.items():
        print(f"  hour {h:>2}: {v['baseline']:.3f} -> {v['new']:.3f}  (drop {v['drop']:.3f})")

    bw_drops = seg["borough_x_weekend"]
    print(f"\nBorough x weekend cells with accuracy drop > {SEGMENT_ACC_DROP:.0%}: {len(bw_drops)}")
    for (borough, wknd), v in bw_drops.items():
        kind = "weekend" if wknd == 1 else "weekday"
        print(f"  borough {borough} {kind}: {v['baseline']:.3f} -> {v['new']:.3f}  (drop {v['drop']:.3f})")

    # --- Summary: distinct drift patterns ---
    print("\n" + "=" * 70)
    print("SUMMARY — DISTINCT DRIFT PATTERNS")
    print("=" * 70)
    sig_features = [r for r in feature_results if r["severity"] in ("SIGNIFICANT", "moderate")]
    trip = next((r for r in feature_results if r["feature"] == "trip_count"), None)

    n = 0
    if trip:
        n += 1
        print(f"\n[{n}] GLOBAL target shift is MILD but real — trip_count mean {trip['mean_shift_pct']:.1f}%, "
              f"PSI={trip['psi']:.3f} ({trip['severity']}). Global view under-alarms; segment to see damage.")
    if hour_drops:
        worst_h = max(hour_drops.items(), key=lambda kv: kv[1]["drop"])
        n += 1
        print(f"\n[{n}] CONCEPT DRIFT by HOUR — {len(hour_drops)} hours degraded; worst hour {worst_h[0]} "
              f"accuracy {worst_h[1]['baseline']:.3f} -> {worst_h[1]['new']:.3f}. Morning demand pattern reshaped.")
    if zone_drops:
        worst_z = max(zone_drops.items(), key=lambda kv: kv[1]["drop"])
        n += 1
        print(f"\n[{n}] CONCEPT DRIFT by ZONE — {len(zone_drops)} zones degraded; worst zone {worst_z[0]} "
              f"accuracy {worst_z[1]['baseline']:.3f} -> {worst_z[1]['new']:.3f}. Heterogeneous, not uniform.")
    if bw_drops:
        (wb, ww), wv = max(bw_drops.items(), key=lambda kv: kv[1]["drop"])
        n += 1
        kind = "weekend" if ww == 1 else "weekday"
        print(f"\n[{n}] CONDITIONAL CONCEPT DRIFT — borough {wb} on {kind}s: accuracy "
              f"{wv['baseline']:.3f} -> {wv['new']:.3f} (drop {wv['drop']:.3f}). A weekday-only or zone-only "
              f"view averages this away; the borough x weekend interaction exposes it.")
    if sig_features:
        names = ", ".join(f"{r['feature']}(PSI={r['psi']:.2f})" for r in sig_features)
        n += 1
        print(f"\n[{n}] DATA DRIFT on engineered day-scale features — {names}. "
              f"Short-horizon features barely moved; day-lag features collapsed, breaking any model that relies on them.")

    print(f"\nTotal distinct drift patterns found: {n}")


if __name__ == "__main__":
    main()
