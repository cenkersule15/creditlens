# CreditLens model card

## Purpose

Estimate next-month payment default risk for existing credit-card customers in the historical UCI Taiwan cohort. Intended use is reproducible portfolio research and interview demonstration. No evidence supports deployment on current German customers or automated credit decisions.

## Development protocol

- Source: 30,000 rows, 23 original features plus ID and binary target; financial model uses 19 features.
- Seed: 42. Financially identical profiles are grouped using a hash of the 19 model inputs before splitting. This keeps even conflicting labels for the same profile in one partition. There are 817 repeated financial profiles after the first occurrence.
- GroupShuffleSplit yields train 16,493 / calibration 4,473 / validation 4,554 / test 4,480 rows. Approximate intended shares: 55 / 15 / 15 / 15 percent. The split is grouped and random, not time-based or exactly stratified.
- Logistic Regression uses StandardScaler fitted on training data only; C=1.0.
- XGBoost compares max_depth 2, 3 and 4, with 250 trees and learning rate 0.04. Remaining parameters are fixed in `train.py`.
- Validation Average Precision selects depth 3. No early stopping or test-driven tuning is used.
- A sigmoid calibrator is fitted on raw XGBoost margins using the separate calibration partition. Positive slope is required. This is a held-out Platt-style calibration.
- A threshold grid from 0 to 1 in steps of 0.01 minimises illustrative validation cost. The cost charges every review and each missed default. Costs are neither estimated from data nor expressed in currency.
- The same validation data supports model and threshold selection; it is not an unbiased estimate of deployment performance. The fixed test partition is reserved for final reporting.

## Explanation semantics

TreeSHAP uses `tree_path_dependent` feature dependence and explains raw XGBoost log-odds. If calibration is `sigmoid(a * margin + b)`, every SHAP contribution is multiplied by `a`, and the base value becomes `a * base + b`. Their sum then exactly reconstructs calibrated log-odds; applying sigmoid reproduces the displayed probability. Tests check this identity. Contributions are not probability percentage points or causal effects. Correlated repayment histories can make individual attributions ambiguous.

Global importance is mean absolute transformed SHAP across 300 training profiles. It is not an attribution over all test customers.

## Evaluation

Full results: `artifacts/report.json`. Calibrated XGBoost test ROC-AUC is 0.7798; AP 0.5511; Brier 0.13320; log loss 0.42462. Raw XGBoost has slightly better Brier (0.13306) and log loss (0.42428). Calibration was not beneficial on these test proper scoring rules and is not claimed to improve reliability. Ranking metrics remain identical under the monotone transform.

At threshold 0.22, the test confusion matrix is TN=2795, FP=719, FN=353, TP=613. Precision is 0.4602 and recall 0.6346. Other models' threshold-dependent metrics use the same numeric threshold for transparency, not a separately optimised policy.

300 row-bootstrap resamples quantify uncertainty conditional on this fixed model and test sample. Because repeated profiles can occur within a test partition, row bootstrapping does not capture their dependence; intervals may be optimistic. It also does not include retraining, hyperparameter-selection or population-shift uncertainty.

Sex and age slices report sample size, defaults, observed risk, mean prediction, review rate, TPR and FPR. These are descriptive, lack subgroup confidence intervals, and are not a fairness certification. Small groups are particularly uncertain.

## Monitoring and operational limits

PSI compares incoming features with train-derived quantile histograms, using clipped proportions. 0.10 and 0.25 are heuristic display thresholds, not statistical guarantees. Quantile-bin PSI can miss shifts within bins, including some degenerate distributions; range checks provide a separate signal. A deliberately shifted demo is synthetic. No claim of observed temporal drift is made.

CSV inputs are limited to 50,000 rows and 15 MB. Additional columns are ignored for model input. NaN, infinities, invalid statuses, non-positive limits and negative payment amounts are rejected. Extreme but finite values may still be unsuitable for inference. The app does not persist uploaded batches to disk.

Only bundled, trusted joblib artifacts should be loaded. Joblib files can execute code and must never be accepted from untrusted users. This local demo has no production authentication, access control, request audit trail or service-level guarantees.

## Follow-up research

External/time-based validation, group-bootstrap uncertainty, richer subgroup intervals, categorical repayment encoding, expanded baselines, domain-specific cost estimation and a genuine longitudinal monitoring dataset would strengthen the project. These are future work, not implemented claims.
