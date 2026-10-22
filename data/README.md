# Dataset attribution and dictionary

Yeh, I. (2009). **Default of Credit Card Clients** [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H

Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

License: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/

The original XLS is redistributed unchanged in `raw/credit_clients.xls`; only the local filename differs. SHA-256: `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933`.

Derived CSVs select rows/columns, convert values to numeric form and append model outputs. Derived distributions, metrics and example profiles originate from this dataset. These changes do not imply endorsement by the dataset author or UCI.

## Model inputs

| Fields | Meaning |
| --- | --- |
| `LIMIT_BAL` | Credit limit, NT$ |
| `PAY_0`, `PAY_2` … `PAY_6` | Repayment status, September back through April 2005 |
| `BILL_AMT1` … `BILL_AMT6` | Statement amounts, September back through April, NT$ |
| `PAY_AMT1` … `PAY_AMT6` | Previous payment amounts, September back through April, NT$ |
| `default payment next month` | Target: 1 = default, 0 = no default |

The source uses unusual repayment codes, including -2, -1 and 0, whose meanings are not fully described by the main UCI dictionary. The project preserves these observed source codes and treats them as numeric inputs rather than inventing definitions. This ordinal treatment is a modelling simplification, especially for Logistic Regression.

Negative statement balances are valid in the source and permitted. Payment amounts must be non-negative. Inputs outside training ranges are reported but not automatically rejected. All amounts remain NT$, without conversion or inflation adjustment.

Excluded predictors: `ID`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`. Audit fields retain the source coding; `SEX`: 1 = male, 2 = female. Removing these columns does not guarantee fair predictions.
