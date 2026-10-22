import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, log_loss, precision_score, recall_score, roc_auc_score


def cost_at_threshold(y, probability, threshold, missed_cost=5.0, review_cost=1.0):
    y = np.asarray(y, dtype=int)
    flagged = np.asarray(probability) >= threshold
    missed = int(((y == 1) & ~flagged).sum())
    reviews = int(flagged.sum())
    return {"threshold": float(threshold), "cost_per_customer": float((missed * missed_cost + reviews * review_cost) / len(y)), "missed_defaults": missed, "reviews": reviews, "review_rate": float(flagged.mean())}


def threshold_curve(y, probability, missed_cost=5.0, review_cost=1.0):
    return pd.DataFrame([cost_at_threshold(y, probability, t, missed_cost, review_cost) for t in np.linspace(0, 1, 101)])


def metrics(y, probability, threshold):
    predicted = np.asarray(probability) >= threshold
    return {"roc_auc": float(roc_auc_score(y, probability)), "average_precision": float(average_precision_score(y, probability)), "brier_score": float(brier_score_loss(y, probability)), "log_loss": float(log_loss(y, probability)), "precision": float(precision_score(y, predicted, zero_division=0)), "recall": float(recall_score(y, predicted, zero_division=0)), "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist()}


def bootstrap_intervals(y, probability, seed=42, rounds=300):
    rng = np.random.default_rng(seed)
    y, probability = np.asarray(y), np.asarray(probability)
    scores = []
    for _ in range(rounds):
        indices = rng.integers(0, len(y), len(y))
        if np.unique(y[indices]).size < 2:
            continue
        scores.append([roc_auc_score(y[indices], probability[indices]), average_precision_score(y[indices], probability[indices])])
    bounds = np.quantile(scores, [.025, .975], axis=0)
    return {name: bounds[:, i].tolist() for i, name in enumerate(["roc_auc", "average_precision"])}


def subgroup_metrics(frame, y, probability, threshold):
    records = []
    slices = {"SEX": frame["SEX"].map({1: "Male", 2: "Female"}), "AGE": pd.cut(frame["AGE"], [0, 29, 44, 59, np.inf], labels=["Under 30", "30–44", "45–59", "60+"])}
    for attribute, values in slices.items():
        for group in values.dropna().unique():
            mask = np.asarray(values == group)
            target, p = np.asarray(y)[mask], np.asarray(probability)[mask]
            pred = p >= threshold
            positives, negatives = target == 1, target == 0
            records.append({"attribute": attribute, "group": str(group), "n": int(mask.sum()), "defaults": int(target.sum()), "default_rate": float(target.mean()), "mean_prediction": float(p.mean()), "review_rate": float(pred.mean()), "true_positive_rate": float(pred[positives].mean()) if positives.any() else None, "false_positive_rate": float(pred[negatives].mean()) if negatives.any() else None})
    return records
