import numpy as np
import pandas as pd


def reference_profile(frame):
    profile = {}
    for column in frame:
        values = frame[column].to_numpy(dtype=float)
        inner = np.unique(np.quantile(values, np.linspace(0, 1, 11)))[1:-1]
        edges = np.r_[-np.inf, inner, np.inf]
        counts = np.histogram(values, bins=edges)[0]
        profile[column] = {"inner_edges": inner.tolist(), "proportions": (counts / len(values)).tolist(), "minimum": float(values.min()), "maximum": float(values.max())}
    return profile


def drift_report(profile, frame):
    records = []
    for column, reference in profile.items():
        values = frame[column].to_numpy(dtype=float)
        edges = np.r_[-np.inf, reference["inner_edges"], np.inf]
        current = np.histogram(values, bins=edges)[0] / len(values)
        p = np.clip(np.asarray(reference["proportions"]), 1e-6, None)
        q = np.clip(current, 1e-6, None)
        p, q = p / p.sum(), q / q.sum()
        psi = float(np.sum((q - p) * np.log(q / p)))
        outside = float(((values < reference["minimum"]) | (values > reference["maximum"])).mean())
        records.append({"feature": column, "psi": psi, "outside_training_range": outside, "signal": "Investigate" if psi >= .25 else "Watch" if psi >= .1 else "Stable"})
    return pd.DataFrame(records).sort_values("psi", ascending=False).reset_index(drop=True)
