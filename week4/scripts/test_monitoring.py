"""
Tests for the Week 4 monitoring stack.

Synthetic data with a known structure (slot-aware demand, a frozen zone_slot_baseline
proxy) lets us assert two things: healthy data raises no alerts, and injected drift /
quality faults are detected by the right metric. Run with: pytest scripts/test_monitoring.py
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric_template import MetricComputer
import compute_metrics as cm
import detect_drift as dd


def make_df(zones=(1, 2, 3, 4, 5), days=3, seed=0, level_scale=1.0, start="2026-01-01"):
    """
    Build a slot-aware demand frame.

    zone_slot_baseline is the FROZEN per-slot level (never scaled). trip_count and the
    day-scale features follow level_scale, so level_scale<1 simulates a demand drop where
    the frozen proxy no longer matches reality (concept drift), and the day-lag feature
    distributions shift (data drift).
    """
    rng = np.random.default_rng(seed)
    n_slots = days * 96
    t0 = pd.Timestamp(start)
    recs = []
    for z in zones:
        for s in range(n_slots):
            ts = t0 + pd.Timedelta(minutes=15 * s)
            hour = ts.hour
            slot_level = 5.0 + 12.0 * max(0.0, np.sin((hour - 3) / 24 * 2 * np.pi))  # day shape
            actual = max(0.0, round(slot_level * level_scale + rng.normal(0, 1.5)))
            recs.append({
                "PULocationID": z,
                "time_bucket": ts,
                "hour": hour,
                "dayofweek": ts.dayofweek,
                "trip_count": int(actual),
                "zone_slot_baseline": slot_level,                 # frozen reference
                "roll_mean_1day": slot_level * level_scale,
                "lag_1day": slot_level * level_scale,
                "lag_1week": slot_level * level_scale,
                "lag_15min": actual,
                "lag_1h": actual,
                "borough_id": 0 if z <= 2 else (1 if z <= 4 else 2),
                "is_weekend": int(ts.dayofweek >= 5),
            })
    return pd.DataFrame(recs)


@pytest.fixture
def baseline():
    return make_df(seed=0, level_scale=1.0)


@pytest.fixture
def healthy_new():
    return make_df(seed=1, level_scale=1.0, start="2026-02-02")


@pytest.fixture
def drifted_new():
    # Demand drops to 55% of baseline -> proxy over-predicts, day-features shift down.
    return make_df(seed=2, level_scale=0.55, start="2026-02-02")


def _pred_act(df):
    return df["zone_slot_baseline"].to_numpy(), df["trip_count"].to_numpy()


# --- Individual metrics: healthy vs faulty ---

def test_accuracy_high_on_healthy(baseline, healthy_new):
    mc = MetricComputer(baseline)
    p, a = _pred_act(healthy_new)
    assert mc.metric_1_accuracy(healthy_new, p, a) > 0.8


def test_accuracy_drops_on_drift(baseline, drifted_new):
    mc = MetricComputer(baseline)
    p_b, a_b = _pred_act(baseline)
    p_d, a_d = _pred_act(drifted_new)
    assert mc.metric_1_accuracy(baseline, p_b, a_b) - mc.metric_1_accuracy(drifted_new, p_d, a_d) > 0.05


def test_accuracy_by_zone_returns_all_zones(baseline):
    mc = MetricComputer(baseline)
    p, a = _pred_act(baseline)
    by_zone = mc.metric_2_accuracy_by_zone(baseline, p, a)
    assert set(by_zone.keys()) == {1, 2, 3, 4, 5}


def test_null_rates_detect_injected_nulls(baseline):
    mc = MetricComputer(baseline)
    assert mc.metric_3_null_rates(baseline)["trip_count"] == 0.0
    bad = baseline.copy()
    bad.loc[bad.index[:50], "trip_count"] = np.nan
    assert mc.metric_3_null_rates(bad)["trip_count"] > 0.0


def test_ks_not_significant_when_same_distribution(baseline, healthy_new):
    mc = MetricComputer(baseline)
    res = mc.metric_4_ks_test(healthy_new)
    assert res["statistic"] < 0.1


def test_ks_significant_on_shifted_feature(baseline, drifted_new):
    mc = MetricComputer(baseline)
    res = mc.metric_4_ks_test(drifted_new)
    assert res["significant"] and res["statistic"] > 0.1


def test_psi_near_zero_when_healthy_and_large_on_drift(baseline, healthy_new, drifted_new):
    mc = MetricComputer(baseline)
    assert mc.metric_5_psi(healthy_new) < 0.1
    assert mc.metric_5_psi(drifted_new) > 0.1


def test_prediction_collapse_detected(baseline):
    mc = MetricComputer(baseline)
    normal = baseline["zone_slot_baseline"].to_numpy()
    assert mc.metric_6_prediction_distribution(normal)["collapsed"] is False
    constant = np.full(len(normal), 7.0)
    assert mc.metric_6_prediction_distribution(constant)["collapsed"] is True


def test_data_freshness_age(baseline):
    mc = MetricComputer(baseline)
    most_recent = pd.to_datetime(baseline["time_bucket"]).max()
    assert mc.metric_7_data_freshness(baseline)["age_minutes"] == 0.0
    stale = mc.metric_7_data_freshness(baseline, now=most_recent + pd.Timedelta(hours=2))
    assert stale["age_minutes"] == pytest.approx(120.0)


def test_duplicate_rate(baseline):
    mc = MetricComputer(baseline)
    assert mc.metric_8_duplicate_rate(baseline)["count"] == 0
    dup = pd.concat([baseline, baseline.iloc[:10]], ignore_index=True)
    assert mc.metric_8_duplicate_rate(dup)["count"] == 10


# --- Alert evaluation (compute_metrics) ---

def test_no_alerts_on_healthy(baseline, healthy_new):
    mc = MetricComputer(baseline)
    bm = mc.compute_all_metrics(baseline, *_pred_act(baseline))
    nm = mc.compute_all_metrics(healthy_new, *_pred_act(healthy_new))
    scan = {f: {"psi": mc.metric_5_psi(healthy_new, feature=f),
                "ks": mc.metric_4_ks_test(healthy_new, feature=f)} for f in ["trip_count"]}
    alerts = cm.evaluate_alerts(bm, nm, scan)
    assert all(a["level"] != "CRITICAL" for a in alerts)


def test_critical_alert_on_drift(baseline, drifted_new):
    mc = MetricComputer(baseline)
    bm = mc.compute_all_metrics(baseline, *_pred_act(baseline))
    nm = mc.compute_all_metrics(drifted_new, *_pred_act(drifted_new))
    scan = {f: {"psi": mc.metric_5_psi(drifted_new, feature=f),
                "ks": mc.metric_4_ks_test(drifted_new, feature=f)} for f in ["trip_count", "roll_mean_1day"]}
    alerts = cm.evaluate_alerts(bm, nm, scan)
    assert any(a["level"] == "CRITICAL" for a in alerts)


# --- Drift detection (detect_drift) ---

def test_detect_feature_drift_flags_shift(baseline, drifted_new):
    res = dd.detect_feature_drift(baseline, drifted_new, "roll_mean_1day")
    assert res["severity"] in ("moderate", "SIGNIFICANT")
    assert res["mean_shift_pct"] < 0


def test_detect_concept_drift_by_segment_finds_degraded(baseline, drifted_new):
    seg = dd.detect_concept_drift_by_segment(baseline, drifted_new)
    assert len(seg["zone"]) > 0  # at least one zone degraded under the demand drop
    assert "borough_x_weekend" in seg  # interaction segment is evaluated
