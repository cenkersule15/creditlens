import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from credit_risk.data import FEATURES, ROOT, TARGET, load_data, split_data, validate_features
from credit_risk.evaluation import cost_at_threshold
from credit_risk.model import calibrated_probability, explain, load_bundle
from credit_risk.monitoring import drift_report, reference_profile


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


@pytest.fixture(scope="module")
def demo():
    return pd.read_csv(ROOT / "artifacts/demo_customers.csv")[FEATURES]


def test_partition_integrity():
    frame = load_data()
    parts, groups = split_data(frame)
    assert sum(len(v) for v in parts.values()) == len(frame)
    for a in parts:
        assert .15 < frame.iloc[parts[a]][TARGET].mean() < .3
        for b in parts:
            if a != b:
                assert not set(parts[a]) & set(parts[b])
                assert not set(groups[parts[a]]) & set(groups[parts[b]])


def test_predictions_ignore_extra_columns_and_order(bundle, demo):
    expected = calibrated_probability(bundle, demo)
    modified = demo[FEATURES[::-1]].assign(SEX=99, AGE=-10, ID=1)
    np.testing.assert_allclose(expected, calibrated_probability(bundle, modified))
    assert np.isfinite(expected).all()
    assert ((expected > 0) & (expected < 1)).all()


@pytest.mark.parametrize("index", [0, 1, 2])
def test_shap_reconstructs_calibrated_probability(bundle, demo, index):
    table, base, probability = explain(bundle, demo.iloc[[index]])
    assert len(table) == len(FEATURES)
    assert expit(base + table.contribution.sum()) == pytest.approx(probability, abs=1e-5)


@pytest.mark.parametrize("field,value", [("LIMIT_BAL", 0), ("PAY_0", 1.5), ("PAY_2", 10), ("PAY_AMT1", -1), ("BILL_AMT1", np.nan), ("LIMIT_BAL", np.inf)])
def test_invalid_input_rejected(demo, field, value):
    invalid = demo.astype(float).copy()
    invalid.loc[0, field] = value
    with pytest.raises(ValueError):
        validate_features(invalid)


def test_negative_bill_is_valid(demo):
    frame = demo.copy()
    frame.loc[0, "BILL_AMT1"] = -500
    assert validate_features(frame).iloc[0].BILL_AMT1 == -500


def test_missing_and_empty_inputs(demo):
    with pytest.raises(ValueError, match="Missing columns"):
        validate_features(demo.drop(columns="LIMIT_BAL"))
    with pytest.raises(ValueError, match="between"):
        validate_features(demo.iloc[:0])


def test_policy_cost_counts_every_review():
    result = cost_at_threshold([1, 0, 1, 0], [.9, .8, .2, .1], .5, missed_cost=5, review_cost=1)
    assert result["reviews"] == 2
    assert result["missed_defaults"] == 1
    assert result["cost_per_customer"] == pytest.approx(7 / 4)


def test_psi_identity_and_shift():
    frame = pd.DataFrame({"x": np.arange(1000)})
    profile = reference_profile(frame)
    assert drift_report(profile, frame).psi.iloc[0] == pytest.approx(0)
    changed = drift_report(profile, frame + 10000)
    assert changed.psi.iloc[0] > .25
    assert changed.outside_training_range.iloc[0] == 1


def test_artifact_predictions_match_runtime(bundle):
    saved = pd.read_csv(ROOT / "artifacts/test_predictions.csv")
    actual = calibrated_probability(bundle, saved)
    np.testing.assert_allclose(actual, saved.probability, atol=1e-7)
    report = json.loads((ROOT / "artifacts/report.json").read_text())
    assert report["duplicate_groups_cross_partitions"] == 0
    assert set(bundle["features"]).isdisjoint(report["excluded_features"])
