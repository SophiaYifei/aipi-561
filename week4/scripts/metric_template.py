"""
Monitoring metrics skeleton.

This file defines 8 metric stubs for monitoring data and model health.
Implement at least 5 of the 8 metrics based on your monitoring framework design.
Each metric should compute a specific health signal about your data/model,
and return a dict (or float) that can be checked against your alert thresholds.
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


# --- Monitoring constants (grounded in the Jan 1-15 baseline; see BASELINE_METRICS.md) ---
# Accuracy proxy: this demand model has no stored prediction column, so we use the frozen
# per-zone-per-slot historical baseline (zone_slot_baseline) as a stand-in for a model
# "trained in January and never updated". It is slot-aware (varies by time-of-day) and stays
# fixed while reality drifts, so a drop in its accuracy is genuine model staleness rather than
# a corrupted-feature artifact. A prediction is "correct" when it lands within a tolerance
# band of the actual count.
PRED_PROXY_COL = "zone_slot_baseline"   # frozen reference used as the naive forecast
ACC_ABS_TOL = 5.0          # absolute tolerance floor (trips) for low-volume slots
ACC_REL_TOL = 0.5          # relative tolerance (50% of the actual count) for high-volume slots

KEY_COLUMNS = ["PULocationID", "time_bucket"]   # uniqueness key (shared with Week 3 validator)
CRITICAL_NULL_COLUMNS = ["trip_count", "PULocationID"]
LAG_NULL_COLUMNS = ["lag_15min", "lag_1h", "lag_1day", "roll_mean_1day"]
DRIFT_FEATURE = "trip_count"                     # primary feature for KS / PSI
FRESHNESS_BUCKET_MINUTES = 15                    # data arrives every 15 minutes
PRED_COLLAPSE_RATIO = 0.5                        # std below 0.5x baseline std => collapse risk


class MetricComputer:
    """Compute monitoring metrics for drift detection."""

    def __init__(self, baseline_df: pd.DataFrame):
        """Initialize with baseline data and pre-compute baseline references."""
        self.baseline_df = baseline_df
        # Baseline distribution of the drift feature (reused by KS test and PSI).
        self.baseline_feature = baseline_df[DRIFT_FEATURE].dropna().values
        # Baseline spread of the naive predictor, used to flag prediction collapse.
        if PRED_PROXY_COL in baseline_df.columns:
            self.baseline_pred_std = float(baseline_df[PRED_PROXY_COL].dropna().std())
        else:
            self.baseline_pred_std = None

    # --- helper: tolerance-band hit mask shared by metric_1 and metric_2 ---
    def _accuracy_hits(self, predictions: np.ndarray, actuals: np.ndarray) -> np.ndarray:
        """Boolean mask of correct predictions under the tolerance band, NaNs excluded."""
        pred = np.asarray(predictions, dtype="float64")
        act = np.asarray(actuals, dtype="float64")
        valid = ~(np.isnan(pred) | np.isnan(act))
        tol = np.maximum(ACC_ABS_TOL, ACC_REL_TOL * np.abs(act))
        hits = np.abs(pred - act) <= tol
        return hits & valid, valid

    def metric_1_accuracy(
        self, new_df: pd.DataFrame, predictions: np.ndarray, actuals: np.ndarray
    ) -> float:
        """
        Metric 1: Overall Accuracy

        Fraction of predictions within the tolerance band of the actual count.
        Baseline ~0.63; alert if it drops >0.05 (warn) / >0.10 (critical) below baseline.
        """
        hits, valid = self._accuracy_hits(predictions, actuals)
        if valid.sum() == 0:
            return float("nan")
        return float(hits.sum() / valid.sum())

    def metric_2_accuracy_by_zone(
        self, new_df: pd.DataFrame, predictions: np.ndarray, actuals: np.ndarray
    ) -> dict:
        """
        Metric 2: Accuracy by Zone

        Per-zone tolerance-band accuracy. Global numbers hide segment failures, so we
        score each PULocationID separately and let the caller flag zones that fell off baseline.
        """
        hits, valid = self._accuracy_hits(predictions, actuals)
        zones = new_df["PULocationID"].to_numpy()
        result = {}
        for z in np.unique(zones):
            mask = (zones == z) & valid
            n = int(mask.sum())
            if n == 0:
                continue
            result[int(z)] = float(hits[mask].sum() / n)
        return result

    def metric_3_null_rates(self, new_df: pd.DataFrame) -> dict:
        """
        Metric 3: Null Rates for Critical Fields

        Null fraction for target/key columns and lag features. Baseline is ~0% for all;
        alert if any critical column exceeds 1% (lag features 2%).
        """
        cols = [c for c in CRITICAL_NULL_COLUMNS + LAG_NULL_COLUMNS if c in new_df.columns]
        return {c: float(new_df[c].isna().mean()) for c in cols}

    def _baseline_values(self, feature: str) -> np.ndarray:
        """Baseline distribution for a feature (cached for the default DRIFT_FEATURE)."""
        if feature == DRIFT_FEATURE:
            return self.baseline_feature
        return self.baseline_df[feature].dropna().values

    def metric_4_ks_test(self, new_df: pd.DataFrame, feature: str = DRIFT_FEATURE) -> dict:
        """
        Metric 4: KS Test for Distribution Shift

        Two-sample Kolmogorov-Smirnov test comparing the new distribution of `feature`
        (trip_count by default) to the baseline. p < 0.05 (warn) / < 0.01 (critical)
        indicates a significant shift. The statistic is the effect size: with large N the
        p-value is almost always tiny, so callers should gate alerts on the statistic too.
        """
        new_feature = new_df[feature].dropna().values
        stat, pvalue = ks_2samp(self._baseline_values(feature), new_feature)
        return {
            "feature": feature,
            "statistic": float(stat),
            "p_value": float(pvalue),
            "significant": bool(pvalue < 0.05),
        }

    def metric_5_psi(self, new_df: pd.DataFrame, bins: int = 10, feature: str = DRIFT_FEATURE) -> float:
        """
        Metric 5: Population Stability Index

        PSI between baseline and new distribution of `feature` (trip_count by default)
        using baseline quantile bins. PSI < 0.10 stable, 0.10-0.25 watch, > 0.25 shift.
        """
        base = self._baseline_values(feature)
        new = new_df[feature].dropna().values
        # Quantile edges from the baseline; unique() guards against ties in skewed counts.
        edges = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return float("nan")
        edges[0], edges[-1] = -np.inf, np.inf
        base_pct = np.histogram(base, bins=edges)[0] / len(base)
        new_pct = np.histogram(new, bins=edges)[0] / len(new)
        base_pct = np.clip(base_pct, 1e-6, None)
        new_pct = np.clip(new_pct, 1e-6, None)
        return float(np.sum((new_pct - base_pct) * np.log(new_pct / base_pct)))

    def metric_6_prediction_distribution(self, predictions: np.ndarray) -> dict:
        """
        Metric 6: Prediction Distribution Shift

        Track mean/std of the (proxy) predictions and flag collapse, i.e. the model
        emitting a near-constant value (std far below the baseline spread).
        """
        pred = pd.Series(predictions, dtype="float64").dropna()
        std = float(pred.std()) if len(pred) else float("nan")
        mean = float(pred.mean()) if len(pred) else float("nan")
        collapsed = False
        if self.baseline_pred_std is not None and not np.isnan(std):
            collapsed = std < PRED_COLLAPSE_RATIO * self.baseline_pred_std
        return {
            "mean": mean,
            "std": std,
            "baseline_std": self.baseline_pred_std,
            "collapsed": bool(collapsed),
        }

    def metric_7_data_freshness(self, new_df: pd.DataFrame, now: pd.Timestamp = None) -> dict:
        """
        Metric 7: Data Freshness

        Age of the most recent record. In production `now` is wall-clock time; here it
        defaults to the newest bucket so the metric reports the internal delivery gap.
        Alert if age exceeds a few 15-minute buckets (warn 30min / critical 60min).
        """
        most_recent = pd.to_datetime(new_df["time_bucket"]).max()
        if now is None:
            now = most_recent
        age_minutes = (pd.Timestamp(now) - most_recent).total_seconds() / 60.0
        return {
            "most_recent": str(most_recent),
            "age_minutes": float(age_minutes),
            "age_hours": float(age_minutes / 60.0),
            "missed_buckets": float(age_minutes / FRESHNESS_BUCKET_MINUTES),
        }

    def metric_8_duplicate_rate(self, new_df: pd.DataFrame) -> dict:
        """
        Metric 8: Duplicate Rate

        Fraction of rows duplicated on the (PULocationID, time_bucket) key. Baseline is 0;
        any duplication points to a broken ingestion/join.
        """
        keys = [c for c in KEY_COLUMNS if c in new_df.columns]
        dup_mask = new_df.duplicated(subset=keys, keep="first")
        count = int(dup_mask.sum())
        return {
            "rate": float(count / len(new_df)) if len(new_df) else 0.0,
            "count": count,
        }

    def compute_all_metrics(
        self,
        new_df: pd.DataFrame,
        predictions: np.ndarray = None,
        actuals: np.ndarray = None,
    ) -> dict:
        """
        Compute all metrics and return a single results dict.

        Performance metrics (1, 2, 6) require predictions/actuals; when those are not
        supplied the metric is skipped so the data-quality and drift metrics still run.
        """
        results = {}
        results["null_rates"] = self.metric_3_null_rates(new_df)
        results["ks_test"] = self.metric_4_ks_test(new_df)
        results["psi"] = self.metric_5_psi(new_df)
        results["data_freshness"] = self.metric_7_data_freshness(new_df)
        results["duplicate_rate"] = self.metric_8_duplicate_rate(new_df)

        if predictions is not None and actuals is not None:
            results["accuracy"] = self.metric_1_accuracy(new_df, predictions, actuals)
            results["accuracy_by_zone"] = self.metric_2_accuracy_by_zone(
                new_df, predictions, actuals
            )
            results["prediction_distribution"] = self.metric_6_prediction_distribution(
                predictions
            )

        return results
