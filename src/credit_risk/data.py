import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
SOURCE_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
SOURCE_SHA256 = "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"
STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENTS = [f"PAY_AMT{i}" for i in range(1, 7)]
FEATURES = ["LIMIT_BAL", *STATUS, *BILLS, *PAYMENTS]
TARGET = "default payment next month"
EXCLUDED = ["ID", "SEX", "EDUCATION", "MARRIAGE", "AGE"]


def acquire(path=None):
    path = Path(path) if path else ROOT / "data/raw/credit_clients.xls"
    if not path.exists():
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "CreditRiskPortfolio/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        names = [n for n in archive.namelist() if n.endswith(".xls")]
        if len(names) != 1:
            raise ValueError("Unexpected UCI archive contents")
        content = archive.read(names[0])
        if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
            raise ValueError("UCI source checksum mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Local source checksum mismatch; restore the original UCI file")
    return path


def load_data():
    frame = pd.read_excel(acquire(), header=1)
    if frame.shape != (30000, 25) or frame[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Unexpected dataset shape or missing values")
    return frame


def validate_features(frame):
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    if frame.empty or len(frame) > 50000:
        raise ValueError("Provide between 1 and 50,000 rows")
    values = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("All model inputs must be finite numbers without missing values")
    if (values["LIMIT_BAL"] <= 0).any():
        raise ValueError("Credit limits must be positive")
    status = values[STATUS].to_numpy()
    if ((status < -2) | (status > 9) | (status != np.floor(status))).any():
        raise ValueError("Repayment status must be an integer between -2 and 9")
    if (values[PAYMENTS] < 0).any().any():
        raise ValueError("Payment amounts must be non-negative; bill balances may be negative")
    return values.astype(float)


def split_data(frame, seed=42):
    groups = pd.util.hash_pandas_object(frame[FEATURES], index=False).to_numpy()
    remaining = np.arange(len(frame))
    parts = {}
    for i, (name, fraction) in enumerate([("test", .15), ("calibration", .15 / .85), ("validation", .15 / .70)]):
        splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed + i)
        keep, holdout = next(splitter.split(remaining, groups=groups[remaining]))
        parts[name] = remaining[holdout]
        remaining = remaining[keep]
    parts["train"] = remaining
    return parts, groups
