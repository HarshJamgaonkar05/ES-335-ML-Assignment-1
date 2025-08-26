import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold

np.random.seed(42)


# Generating the dataset

X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2,
    random_state=1, n_clusters_per_class=2, class_sep=0.5
)

# First, we convert the dataset to DataFrame/Series for our DecisionTree
X_df = pd.DataFrame(X, columns=["x0", "x1"])
y_ser = pd.Series(y, dtype="category")

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Synthetic classification dataset")
plt.show()



# Q2(a): Train/test split 70/30 to find the accuracy, precision and recall

idx = np.arange(len(X_df))
np.random.shuffle(idx)
split_idx = int(0.7 * len(idx))
train_idx, test_idx = idx[:split_idx], idx[split_idx:]
X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
y_train, y_test = y_ser.iloc[train_idx], y_ser.iloc[test_idx]

tree = DecisionTree(criterion="information_gain", max_depth=5)
tree.fit(X_train, y_train)
y_hat = tree.predict(X_test)

print("=== Q2 (a) Results ===")
print("Accuracy:", accuracy(y_hat, y_test))
for cls in sorted(y_ser.unique()):
    print(f"Class {cls} Precision:", precision(y_hat, y_test, cls))
    print(f"Class {cls} Recall   :", recall(y_hat, y_test, cls))


# Q2 (b): 5-fold cross-validation with nested cross validation to find the optimum depth of the tree

def cross_val_score_for_depth(X, y, depth, n_splits=5):
    """
    Return average accuracy across folds for a given max_depth.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        tree = DecisionTree(criterion="information_gain", max_depth=depth)
        tree.fit(X_tr, y_tr)
        y_pred = tree.predict(X_val)
        accs.append(accuracy(y_pred, y_val))
    return np.mean(accs)

depths = [1, 2, 3, 4, 5, 6, 7, 8]
cv_scores = []
for d in depths:
    score = cross_val_score_for_depth(X_df, y_ser, depth=d, n_splits=5)
    cv_scores.append(score)
    print(f"Depth {d}: mean CV accuracy = {score:.3f}")

best_depth = depths[np.argmax(cv_scores)]
print("\n=== Q2 (b) Results ===")
print("Best depth found by nested CV:", best_depth)

final_tree = DecisionTree(criterion="information_gain", max_depth=best_depth)
final_tree.fit(X_df, y_ser)
print("Tree structure at best depth:")
final_tree.plot()

# Plot Cross Validation results
plt.figure()
plt.plot(depths, cv_scores, marker="o")
plt.xlabel("max_depth")
plt.ylabel("mean CV accuracy")
plt.title("Nested CV: depth vs accuracy")
plt.grid(True)
plt.show()
