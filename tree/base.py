"""
Decision Tree supporting:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd

from tree.utils import (
    one_hot_encoding,   # Converts categorical features into numeric dummy variables
    check_ifreal,       # Detects if the target y is regression or classification
    opt_split_attribute # Finds the best feature/threshold split
)

np.random.seed(42)


# --------------------------- Node container ---------------------------
class _Node:
    """
    Internal node/leaf in the decision tree.
    Stores info about feature/threshold, children, or prediction.
    """
    def __init__(
        self,
        depth: int,
        is_leaf: bool,
        prediction: Any = None,              # Value at leaf (class or mean for regression)
        feature: Optional[str] = None,       # Which feature is used to split
        threshold: Optional[float] = None,   # Threshold for numeric split
        children: Optional[Dict[Any, "_Node"]] = None,  # For categorical splits
        left: Optional["_Node"] = None,      # Left child for <= threshold
        right: Optional["_Node"] = None,     # Right child for > threshold
    ):
        self.depth = depth
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature = feature
        self.threshold = threshold
        self.children = children
        self.left = left
        self.right = right


# ------------------------------ Tree ---------------------------------
@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # Split rule for classification
    max_depth: int = 5                                    # Maximum depth allowed
    one_hot_categoricals: bool = True                     # If True, one-hot encode categorical inputs

    def __init__(self, criterion: Literal["information_gain", "gini_index"],
                 max_depth: int = 5, one_hot_categoricals: bool = True):
        self.criterion = criterion
        self.max_depth = max_depth
        self.one_hot_categoricals = one_hot_categoricals

        # Will be set later during training
        self._root: Optional[_Node] = None
        self._task: Optional[Literal["classification", "regression"]] = None
        self.classes_: Optional[np.ndarray] = None
        self._train_cols: Optional[pd.Index] = None  # Save training columns after OHE

    # ----------------------------- Public API -----------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the decision tree.
        """
        X = X.copy(); y = y.copy()
        assert len(X) == len(y), "X and y must have same length."

        # Convert categoricals → numeric columns if enabled
        if self.one_hot_categoricals:
            X = one_hot_encoding(X)

        self._train_cols = X.columns  # Remember column order

        # Detect whether it's regression (real output) or classification (discrete)
        self._task = "regression" if check_ifreal(y) else "classification"
        if self._task == "classification":
            # Keep consistent ordering of classes for printing/prediction
            self.classes_ = np.array(sorted(y.unique(), key=lambda z: str(z)))

        # Build the tree recursively
        self._root = self._grow(X, y, depth=0)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict outputs for given inputs."""
        assert self._root is not None, "Call fit() before predict()."
        X = X.copy()

        # Apply same preprocessing as training
        if self.one_hot_categoricals:
            X = one_hot_encoding(X)
            X = X.reindex(columns=self._train_cols, fill_value=0)  # Ensure same cols as training

        preds = []
        for _, row in X.iterrows():
            preds.append(self._predict_row(row, self._root))
        return pd.Series(preds, index=X.index)

    def plot(self) -> None:
        """Pretty-print the tree structure as text."""
        assert self._root is not None, "Call fit() before plot()."
        lines: list[str] = []
        self._render(self._root, prefix="", lines=lines)
        print("\n".join(lines))

    # ------------------------- Recursive builder -------------------------
    def _grow(self, X: pd.DataFrame, y: pd.Series, depth: int) -> _Node:
        # Stop if: depth too large, only one class/constant output left, or no features
        if depth >= self.max_depth or y.nunique(dropna=False) <= 1 or X.shape[0] == 0 or X.shape[1] == 0:
            return _Node(depth=depth, is_leaf=True, prediction=self._leaf_value(y))

        # Pick best split using utils.opt_split_attribute
        best = opt_split_attribute(X, y, self.criterion, features=pd.Series(X.columns))
        if best is None:  # No good split found
            return _Node(depth=depth, is_leaf=True, prediction=self._leaf_value(y))

        feature = best["feature"]

        if best["kind"] == "numeric":
            thr = float(best["threshold"])
            mask = X[feature] <= thr

            # If split puts all samples on one side → stop
            n_left = int(mask.sum())
            if n_left == 0 or n_left == len(X):
                return _Node(depth=depth, is_leaf=True, prediction=self._leaf_value(y))

            # Recursively build left and right children
            left = self._grow(X[mask], y[mask], depth + 1)
            right = self._grow(X[~mask], y[~mask], depth + 1)
            return _Node(depth=depth, is_leaf=False, feature=feature, threshold=thr,
                         left=left, right=right)

        # If categorical (rare with OHE on): build one child per value
        children: Dict[Any, _Node] = {}
        for v, idx in X[feature].groupby(X[feature]).groups.items():
            Xv, yv = X.loc[idx], y.loc[idx]
            children[v] = self._grow(Xv, yv, depth + 1)
        return _Node(depth=depth, is_leaf=False, feature=feature, children=children)

    # ----------------------------- Utilities -----------------------------
    def _leaf_value(self, y: pd.Series):
        # Leaf stores average value (regression) or most common class (classification)
        if self._task == "regression":
            return float(y.mean()) if len(y) else 0.0
        counts = y.value_counts()
        winners = counts[counts == counts.max()].index
        return sorted(winners, key=lambda z: str(z))[0]  # Tie-break by label name

    def _predict_row(self, row: pd.Series, node: _Node):
        # If we are at a leaf → return stored prediction
        if node.is_leaf:
            return node.prediction

        # If numeric split
        if node.threshold is not None:
            val = row.get(node.feature)

            # If missing value → send to bigger subtree
            if val is None or (isinstance(val, float) and np.isnan(val)):
                go_left = self._size(node.left) >= self._size(node.right)
            else:
                go_left = float(val) <= float(node.threshold)

            return self._predict_row(row, node.left if go_left else node.right)

        # If categorical split
        if node.children is not None:
            val = row.get(node.feature)
            child = node.children.get(val)
            if child is None:
                # If unseen category → fallback to biggest subtree
                sizes = {k: self._size(v) for k, v in node.children.items()}
                child = node.children[max(sizes, key=sizes.get)]
            return self._predict_row(row, child)

        # Fallback (shouldn’t happen)
        return node.prediction

    def _size(self, node: Optional[_Node]) -> int:
        """Helper to count size of subtree (used for NaN/unseen category fallback)."""
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        n = 1
        if node.left is not None:
            n += self._size(node.left)
        if node.right is not None:
            n += self._size(node.right)
        if node.children is not None:
            for ch in node.children.values():
                n += self._size(ch)
        return n

    def _render(self, node: _Node, prefix: str, lines: list[str]):
        """Helper to pretty-print the tree recursively."""
        indent = "    " * node.depth
        if node.is_leaf:
            if self._task == "classification":
                lines.append(f"{indent}{prefix}Class {node.prediction}")
            else:
                lines.append(f"{indent}{prefix}Value {node.prediction:.3f}")
            return

        if node.threshold is not None:
            # Numeric split → print threshold
            lines.append(f"{indent}{prefix}?( {node.feature} <= {node.threshold:.4g} )")
            self._render(node.left,  "Y: ", lines)
            self._render(node.right, "N: ", lines)
        else:
            # Categorical split → print each branch
            lines.append(f"{indent}{prefix}?( {node.feature} ∈ {{...}} )")
            for k in sorted(node.children.keys(), key=lambda z: str(z)):
                self._render(node.children[k], f"{k}: ", lines)

