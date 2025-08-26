"""
Helper utilities for the Decision Tree implementation.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# --------- basic helpers ---------
def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical columns into binary (0/1) dummy variables.
    Keeps numeric columns as they are.
    """
    return pd.get_dummies(X, drop_first=False)


def check_ifreal(y: pd.Series, max_discrete_classes: int = 15) -> bool:
    """
    Decide if target y is regression (real) or classification (discrete).
    - Non-numeric → classification
    - Float dtype → regression
    - Integer dtype:
        * Few unique values (<= max_discrete_classes) → classification
        * Many unique values → regression
    """
    if not pd.api.types.is_numeric_dtype(y):
        return False  # categorical/object => classification
    if pd.api.types.is_float_dtype(y):
        return True   # float => regression
    n_unique = int(y.nunique(dropna=False))
    return False if n_unique <= max_discrete_classes else True


# --------- impurity measures ---------
def entropy(Y: pd.Series) -> float:
    """Shannon entropy: measures uncertainty in labels."""
    if len(Y) == 0:
        return 0.0
    p = Y.value_counts(normalize=True).values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -p * np.log2(p, where=(p > 0))
    h[~np.isfinite(h)] = 0.0
    return float(h.sum())


def gini_index(Y: pd.Series) -> float:
    """Gini impurity: probability of misclassification."""
    if len(Y) == 0:
        return 0.0
    p = Y.value_counts(normalize=True).values.astype(float)
    return float(1.0 - np.sum(p ** 2))


def mse(Y: pd.Series) -> float:
    """Mean squared error: variance around mean (for regression)."""
    if len(Y) == 0:
        return 0.0
    mu = float(Y.mean())
    return float(((Y - mu) ** 2).mean())


# --------- information gain / reduction ---------
def information_gain(Y: pd.Series, split_labels: pd.Series, criterion: str) -> float:
    """
    Calculate how much "impurity" is reduced after a split.
    - For regression: uses MSE
    - For classification: uses entropy or gini
    """
    split_labels = split_labels.reindex(Y.index).fillna("__NAN__SPLIT__")
    N = len(Y)
    if N == 0:
        return 0.0

    if criterion == "mse":  # regression
        parent = mse(Y)
        child_imp = 0.0
        for _, idx in split_labels.groupby(split_labels, sort=False).groups.items():
            Yi = Y.loc[idx]
            child_imp += (len(Yi) / N) * mse(Yi)
        return parent - child_imp

    # classification
    parent = entropy(Y) if criterion == "entropy" else gini_index(Y)
    child_imp = 0.0
    for _, idx in split_labels.groupby(split_labels, sort=False).groups.items():
        Yi = Y.loc[idx]
        imp = entropy(Yi) if criterion == "entropy" else gini_index(Yi)
        child_imp += (len(Yi) / N) * imp
    return parent - child_imp


# --------- attribute search ---------
def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion: str, features: pd.Series):
    """
    Find the best feature (and threshold if numeric) to split on.
    - Numeric: test midpoints between sorted unique values
    - Categorical: split by unique categories
    Returns best feature and gain, or None if no good split.
    """
    task_is_regression = check_ifreal(y)
    crit = "mse" if task_is_regression else ("entropy" if criterion == "information_gain" else "gini")

    best, best_gain = None, 1e-12

    for feature in features:
        s = X[feature]

        if pd.api.types.is_numeric_dtype(s):  # numeric feature
            arr = s.astype(float).values
            arr = arr[~np.isnan(arr)]
            if arr.size <= 1:
                continue
            vals = np.unique(arr)
            if vals.size <= 1:
                continue
            thr_cands = (vals[:-1] + vals[1:]) / 2.0  # midpoints
            col = s.values
            for t in thr_cands:
                split = pd.Series(np.where(col <= t, 0, 1), index=X.index)
                gain = information_gain(y, split, crit)
                if gain > best_gain:
                    best_gain = gain
                    best = {"feature": feature, "kind": "numeric", "threshold": float(t), "gain": float(gain)}

        else:  # categorical feature
            if s.nunique(dropna=False) <= 1:
                continue
            split = s.astype("object").fillna("__NAN__CAT__")
            gain = information_gain(y, split, crit)
            if gain > best_gain:
                best_gain = gain
                best = {"feature": feature, "kind": "categorical", "threshold": None, "gain": float(gain)}

    return best


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Actually split dataset on chosen feature:
    - If numeric: split <= threshold vs > threshold
    - If categorical: split equals value vs not
    """
    s = X[attribute]
    if pd.api.types.is_numeric_dtype(s) and isinstance(value, (int, float, np.floating)):
        mask = s <= float(value)
        return (X[mask], y[mask]), (X[~mask], y[~mask])
    else:
        mask = s.astype("object") == value
        return X[mask], y[mask]


