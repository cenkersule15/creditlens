import joblib
import numpy as np
import pandas as pd
import shap
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from credit_risk.data import FEATURES, ROOT, validate_features


def fit_calibrator(model, features, target):
    margin = model.predict(features, output_margin=True)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(margin.reshape(-1, 1), target)
    if calibrator.coef_[0, 0] <= 0:
        raise ValueError("Calibration is not monotonic; inspect model and data")
    return calibrator


def calibrated_probability(bundle, frame):
    features = validate_features(frame)
    margin = bundle["model"].predict(features, output_margin=True)
    return bundle["calibrator"].predict_proba(margin.reshape(-1, 1))[:, 1]


def explain(bundle, frame):
    features = validate_features(frame)
    if len(features) != 1:
        raise ValueError("Explain one customer at a time")
    explanation = shap.TreeExplainer(bundle["model"], feature_perturbation="tree_path_dependent")(features)
    slope = float(bundle["calibrator"].coef_[0, 0])
    intercept = float(bundle["calibrator"].intercept_[0])
    contributions = explanation.values[0] * slope
    base = float(explanation.base_values[0]) * slope + intercept
    probability = float(expit(base + contributions.sum()))
    expected = float(calibrated_probability(bundle, features)[0])
    if not np.isclose(probability, expected, atol=1e-5):
        raise ValueError("Explanation does not reconstruct the calibrated probability")
    table = pd.DataFrame({"feature": FEATURES, "value": features.iloc[0].values, "contribution": contributions})
    return table, base, expected


def load_bundle(path=None):
    path = path or ROOT / "artifacts/model.joblib"
    return joblib.load(path)
