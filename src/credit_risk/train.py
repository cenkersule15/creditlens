import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from credit_risk.data import FEATURES, ROOT, SOURCE_SHA256, TARGET, load_data, split_data
from credit_risk.evaluation import bootstrap_intervals, cost_at_threshold, metrics, subgroup_metrics, threshold_curve
from credit_risk.model import calibrated_probability, explain, fit_calibrator
from credit_risk.monitoring import reference_profile


def train(output=None):
    output = Path(output or ROOT / "artifacts")
    output.mkdir(parents=True, exist_ok=True)
    frame = load_data()
    splits, groups = split_data(frame)
    x = frame[FEATURES].astype(float)
    y = frame[TARGET]
    tr, ca, va, te = [splits[n] for n in ["train", "calibration", "validation", "test"]]
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=42))
    baseline.fit(x.iloc[tr], y.iloc[tr])
    candidates, scores = [], []
    for depth, trees in [(2, 250), (3, 250), (4, 250)]:
        candidate = XGBClassifier(n_estimators=trees, max_depth=depth, learning_rate=.04, min_child_weight=20, subsample=.85, colsample_bytree=.9, reg_lambda=5, tree_method="hist", eval_metric="logloss", n_jobs=2, random_state=42)
        candidate.fit(x.iloc[tr], y.iloc[tr])
        score = float(average_precision_score(y.iloc[va], candidate.predict_proba(x.iloc[va])[:, 1]))
        candidates.append(candidate)
        scores.append({"max_depth": depth, "n_estimators": trees, "validation_average_precision": score})
    best_index = int(np.argmax([s["validation_average_precision"] for s in scores]))
    model = candidates[best_index]
    bundle = {"model": model, "calibrator": fit_calibrator(model, x.iloc[ca], y.iloc[ca]), "features": FEATURES, "version": "0.1.0"}
    validation_probability = calibrated_probability(bundle, x.iloc[va])
    validation = frame.iloc[va][[TARGET]].copy()
    validation["probability"] = validation_probability
    validation.to_csv(output / "validation_predictions.csv", index=False)
    curve = threshold_curve(y.iloc[va], validation_probability)
    threshold = float(curve.loc[curve.cost_per_customer.idxmin(), "threshold"])
    bundle["threshold"] = threshold
    probability = calibrated_probability(bundle, x.iloc[te])
    raw = model.predict_proba(x.iloc[te])[:, 1]
    lr = baseline.predict_proba(x.iloc[te])[:, 1]
    constant = np.full(len(te), float(y.iloc[tr].mean()))
    comparison = {name: metrics(y.iloc[te], p, threshold) for name, p in [("Prevalence baseline", constant), ("Logistic Regression", lr), ("XGBoost raw", raw), ("XGBoost calibrated", probability)]}
    test = frame.iloc[te].copy()
    test["probability"] = probability
    test["raw_probability"] = raw
    test["logistic_probability"] = lr
    test.to_csv(output / "test_predictions.csv", index=False)
    curve.to_csv(output / "validation_thresholds.csv", index=False)
    sample = x.iloc[tr].sample(n=300, random_state=42)
    import shap
    local_values = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")(sample).values
    importance = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(local_values).mean(axis=0) * float(bundle["calibrator"].coef_[0, 0])}).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(output / "importance.csv", index=False)
    profile = reference_profile(x.iloc[tr])
    (output / "reference_profile.json").write_text(json.dumps(profile, indent=2))
    selected = []
    for q in [.1, .5, .9]:
        selected.append(int(np.argmin(np.abs(probability - np.quantile(probability, q)))))
    demo = test.iloc[selected][["ID", *FEATURES]]
    demo.to_csv(output / "demo_customers.csv", index=False)
    x.iloc[te[:10]].to_csv(ROOT / "data/batch_example.csv", index=False)
    for a in splits:
        for b in splits:
            if a < b and set(groups[splits[a]]) & set(groups[splits[b]]):
                raise ValueError("Feature-identical customer profiles leaked across partitions")
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": SOURCE_SHA256, "seed": 42, "splits": {k: {"rows": len(v), "default_rate": float(y.iloc[v].mean())} for k, v in splits.items()}, "excluded_features": ["ID", "SEX", "EDUCATION", "MARRIAGE", "AGE"], "duplicate_financial_profiles": int(pd.Series(groups).duplicated().sum()), "duplicate_groups_cross_partitions": 0, "candidates": scores, "selected_candidate": scores[best_index], "threshold": threshold, "cost_assumptions": {"missed_default": 5, "each_review": 1, "units": "illustrative cost units; not currency"}, "test_metrics": comparison, "confidence_intervals_95": bootstrap_intervals(y.iloc[te], probability), "test_policy": cost_at_threshold(y.iloc[te], probability, threshold), "subgroups": subgroup_metrics(test, y.iloc[te], probability, threshold), "calibration": {"slope": float(bundle["calibrator"].coef_[0, 0]), "intercept": float(bundle["calibrator"].intercept_[0])}, "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "xgboost", "shap", "streamlit"]}}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    (output / "splits.json").write_text(json.dumps({k: frame.iloc[v].ID.tolist() for k, v in splits.items()}))
    joblib.dump(bundle, output / "model.joblib")
    joblib.dump(baseline, output / "baseline.joblib")
    model.save_model(output / "xgboost.json")
    explain(bundle, demo[FEATURES].iloc[:1])
    print(json.dumps({"test": comparison["XGBoost calibrated"], "threshold": threshold, "splits": report["splits"]}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    train(args.output)
