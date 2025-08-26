"""
Usage script for DecisionTree.
Covers all 4 cases:
> real input, real output
> real input, discrete output
> discrete input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
from tree.base import DecisionTree
from metrics import accuracy, precision, recall, rmse, mae

np.random.seed(42)

# Case 1: Real input, Real output (Regression)

print("\n Case 1: Real Input, Real Output ")
X = pd.DataFrame(np.random.randn(30, 5))
y = pd.Series(np.random.randn(30))
for crit in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=crit)
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print(f"Criteria: {crit}")
    print(f"RMSE: {rmse(y_hat, y):.3f}, MAE: {mae(y_hat, y):.3f}\n")


# Case 2: Real input, Discrete output (Classification)

print("\n Case 2: Real Input, Discrete Output")
X = pd.DataFrame(np.random.randn(30, 5))
y = pd.Series(np.random.randint(5, size=30), dtype="category")
for crit in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=crit)
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print(f"Criteria: {crit}")
    print("Accuracy:", accuracy(y_hat, y))
    for cls in y.unique():
        print(f"Class {cls}: Precision={precision(y_hat, y, cls):.3f}, Recall={recall(y_hat, y, cls):.3f}")
    print()


# Case 3: Discrete input, Discrete output

print("\n Case 3: Discrete Input, Discrete Output")
X = pd.DataFrame({i: pd.Series(np.random.randint(5, size=30), dtype="category") for i in range(5)})
y = pd.Series(np.random.randint(5, size=30), dtype="category")
for crit in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=crit)
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print(f"Criteria: {crit}")
    print("Accuracy:", accuracy(y_hat, y))
    for cls in y.unique():
        print(f"Class {cls}: Precision={precision(y_hat, y, cls):.3f}, Recall={recall(y_hat, y, cls):.3f}")
    print()


# Case 4: Discrete input, Real output (Regression)

print("\n Case 4: Discrete Input, Real Output")
X = pd.DataFrame({i: pd.Series(np.random.randint(5, size=30), dtype="category") for i in range(5)})
y = pd.Series(np.random.randn(30))
for crit in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=crit)
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print(f"Criteria: {crit}")
    print(f"RMSE: {rmse(y_hat, y):.3f}, MAE: {mae(y_hat, y):.3f}\n")
