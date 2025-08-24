import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tree.base import DecisionTree        # your implementation
from metrics import rmse, mae             # your metrics

# For a fair baseline comparison
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold

np.random.seed(42)

# ---------------------------
# 1) Load & clean UCI Auto MPG
# ---------------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
cols = ["mpg", "cylinders", "displacement", "horsepower", "weight",
        "acceleration", "model year", "origin", "car name"]
data = pd.read_csv(url, delim_whitespace=True, header=None, names=cols)

# Drop non-predictive text column
data = data.drop(columns=["car name"])

# horsepower has '?' -> coerce to NaN, then drop rows with NaNs
data["horsepower"] = pd.to_numeric(data["horsepower"], errors="coerce")
data = data.dropna().reset_index(drop=True)

# Targets and features
y = data["mpg"].astype(float)
X = data.drop(columns=["mpg"])

# ---------------------------
# 2) Train / Test split (70/30)
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# ---------------------------
# 3a) Train YOUR DecisionTree (regression via MSE reduction)
# ---------------------------
my_tree = DecisionTree(criterion="information_gain", max_depth=6)  # criterion ignored for regression
my_tree.fit(X_train, y_train)
y_pred_my = my_tree.predict(X_test)

print("=== Automotive Efficiency: DecisionTree ===")
print(f"RMSE: {rmse(y_pred_my, y_test):.3f}")
print(f"MAE : {mae(y_pred_my, y_test):.3f}")
print("\nTree structure (truncated view):")
my_tree.plot()

# ---------------------------
# 3b) Train scikit-learn DecisionTreeRegressor (baseline)
# ---------------------------
sk_tree = DecisionTreeRegressor(criterion="squared_error", max_depth=6, random_state=42)
sk_tree.fit(X_train, y_train)
y_pred_sk = pd.Series(sk_tree.predict(X_test), index=y_test.index)

print("\n=== Automotive Efficiency: scikit-learn DecisionTreeRegressor ===")
print(f"RMSE: {rmse(y_pred_sk, y_test):.3f}")
print(f"MAE : {mae(y_pred_sk, y_test):.3f}")

# ---------------------------
# 4) Cross-validation to tune max_depth
# ---------------------------
def cv_score_for_depth(X, y, depth, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmses = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        tree = DecisionTree(criterion="information_gain", max_depth=depth)
        tree.fit(X_tr, y_tr)
        y_pred = tree.predict(X_val)
        rmses.append(rmse(y_pred, y_val))
    return np.mean(rmses)

depths = list(range(1, 11))
cv_scores = []
for d in depths:
    score = cv_score_for_depth(X, y, d, n_splits=5)
    cv_scores.append(score)
    print(f"Depth {d}: mean CV RMSE = {score:.3f}")

best_depth = depths[np.argmin(cv_scores)]
print("\n=== Cross-validation Results ===")
print("Best depth (lowest RMSE):", best_depth)

# Retrain on full dataset at best depth
final_tree = DecisionTree(criterion="information_gain", max_depth=best_depth)
final_tree.fit(X, y)
print("Tree structure at best depth:")
final_tree.plot()

# Plot CV results
plt.figure()
plt.plot(depths, cv_scores, marker="o")
plt.xlabel("max_depth")
plt.ylabel("mean CV RMSE")
plt.title("Cross-validation: depth vs RMSE (lower is better)")
plt.grid(True)
plt.show()

# ---------------------------
# 5) Short textual comparison (printed)
# ---------------------------
rmse_my, mae_my = rmse(y_pred_my, y_test), mae(y_pred_my, y_test)
rmse_sk, mae_sk = rmse(y_pred_sk, y_test), mae(y_pred_sk, y_test)

print("\n=== Summary ===")
print(f"Your DT   -> RMSE: {rmse_my:.3f}, MAE: {mae_my:.3f}")
print(f"sklearn   -> RMSE: {rmse_sk:.3f}, MAE: {mae_sk:.3f}")
if rmse_my <= rmse_sk:
    print("Our Decision tree matches or slightly beats sklearn on this split.")
else:
    print("sklearn edges out out tree on this split (common with optimized implementations).")

