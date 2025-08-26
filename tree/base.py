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
    one_hot_encoding,   # This converts the categorical features into numeric dummy variables
    check_ifreal,       # This will detect if the target variable y is regression or classification
    opt_split_attribute # This functions helps to find the best feature or the threshold variable
)

np.random.seed(42)


class _Node:
    """
    Internal node/leaf in the decision tree.
    Stores info about feature/threshold, children, or prediction.
    """
    def __init__(
        self,
        depth: int,
        is_leaf: bool,
        prediction: Any = None,              # Tells what value is at leaf (class or mean for regression)
        feature: Optional[str] = None,       # Tells which feature is used to split
        threshold: Optional[float] = None,   # Tells about the threshold for numeric split
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


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # This is the split rule used for classification
    max_depth: int = 5                                    # The maximum depth allowed for the tree
    one_hot_categoricals: bool = True                    

    def __init__(self, criterion: Literal["information_gain", "gini_index"],
                 max_depth: int = 5, one_hot_categoricals: bool = True):
        self.criterion = criterion
        self.max_depth = max_depth
        self.one_hot_categoricals = one_hot_categoricals


        self._root: Optional[_Node] = None
        self._task: Optional[Literal["classification", "regression"]] = None
        self.classes_: Optional[np.ndarray] = None
        self._train_cols: Optional[pd.Index] = None 


    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the decision tree.
        """
        X = X.copy(); y = y.copy()
        assert len(X) == len(y), "X and y must have same length."

        # This will convert the categories to numeric columns
        if self.one_hot_categoricals:
            X = one_hot_encoding(X)

        self._train_cols = X.columns  # To Remember column order

        # This will detect whether it's regression (real output) or classification (discrete)
        self._task = "regression" if check_ifreal(y) else "classification"
        if self._task == "classification":
            self.classes_ = np.array(sorted(y.unique(), key=lambda z: str(z)))

        self._root = self._grow(X, y, depth=0)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict outputs for given inputs."""
        assert self._root is not None, "Call fit() before predict()."
        X = X.copy()

        # Apply same preprocessing as training
        if self.one_hot_categoricals:
            X = one_hot_encoding(X)
            X = X.reindex(columns=self._train_cols, fill_value=0)

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


    def _grow(self, X: pd.DataFrame, y: pd.Series, depth: int) -> _Node:
        # Will stop growing the tree if depth too large, only one class/constant output left or no features
        if depth >= self.max_depth or y.nunique(dropna=False) <= 1 or X.shape[0] == 0 or X.shape[1] == 0:
            return _Node(depth=depth, is_leaf=True, prediction=self._leaf_value(y))

        # This is used to pick up the best feature using utils.opt_split_attribute
        best = opt_split_attribute(X, y, self.criterion, features=pd.Series(X.columns))
        if best is None: 
            return _Node(depth=depth, is_leaf=True, prediction=self._leaf_value(y))

        feature = best["feature"]

        if best["kind"] == "numeric":
            thr = float(best["threshold"])
            mask = X[feature] <= thr

            # If split puts all samples on one side then it will stop
            n_left = int(mask.sum())
            if n_left == 0 or n_left == len(X):
                return _Node(depth=depth, is_leaf=True, prediction=self._leaf_value(y))

            # Recursively build left and right children
            left = self._grow(X[mask], y[mask], depth + 1)
            right = self._grow(X[~mask], y[~mask], depth + 1)
            return _Node(depth=depth, is_leaf=False, feature=feature, threshold=thr,
                         left=left, right=right)

        children: Dict[Any, _Node] = {}
        for v, idx in X[feature].groupby(X[feature]).groups.items():
            Xv, yv = X.loc[idx], y.loc[idx]
            children[v] = self._grow(Xv, yv, depth + 1)
        return _Node(depth=depth, is_leaf=False, feature=feature, children=children)


    def _leaf_value(self, y: pd.Series):
        # Leaf stores average value showing regression or most common class showing classification
        if self._task == "regression":
            return float(y.mean()) if len(y) else 0.0
        counts = y.value_counts()
        winners = counts[counts == counts.max()].index
        return sorted(winners, key=lambda z: str(z))[0] 

    def _predict_row(self, row: pd.Series, node: _Node):
        # If we are at a leaf then it will return the stored prediction
        if node.is_leaf:
            return node.prediction

        # If there is a numeric split
        if node.threshold is not None:
            val = row.get(node.feature)

            # If there is a missing value then we send it to bigger subtree
            if val is None or (isinstance(val, float) and np.isnan(val)):
                go_left = self._size(node.left) >= self._size(node.right)
            else:
                go_left = float(val) <= float(node.threshold)

            return self._predict_row(row, node.left if go_left else node.right)

        # If there is a categorical split
        if node.children is not None:
            val = row.get(node.feature)
            child = node.children.get(val)
            if child is None:
                # If there is an unseen category it will fallback to biggest subtree
                sizes = {k: self._size(v) for k, v in node.children.items()}
                child = node.children[max(sizes, key=sizes.get)]
            return self._predict_row(row, child)

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
            # If there is a numeric split it will print the threshold value
            lines.append(f"{indent}{prefix}?( {node.feature} <= {node.threshold:.4g} )")
            self._render(node.left,  "Y: ", lines)
            self._render(node.right, "N: ", lines)
        else:
            # If there is a categorical split it will print the branch
            lines.append(f"{indent}{prefix}?( {node.feature} ∈ {{...}} )")
            for k in sorted(node.children.keys(), key=lambda z: str(z)):
                self._render(node.children[k], f"{k}: ", lines)



