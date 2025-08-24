import time
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100  # number of repetitions to average runtime

# -----------------------
# 1) DATA GENERATORS
# -----------------------
def make_data_discrete_features(N: int, M: int, K: int = 2) -> pd.DataFrame:
    """Generate binary (0/1) features with shape (N, M)."""
    X = pd.DataFrame(np.random.randint(0, 2, size=(N, M)),
                     columns=[f"f{i}" for i in range(M)])
    return X

def make_data_real_features(N: int, M: int) -> pd.DataFrame:
    """Generate real-valued features from N(0,1)."""
    X = pd.DataFrame(np.random.randn(N, M),
                     columns=[f"x{i}" for i in range(M)])
    return X

def make_targets_discrete_from_binary(X: pd.DataFrame, K: int = 2) -> pd.Series:
    """Generate classification labels from binary features."""
    N, M = X.shape
    w = np.array([2.0, -1.5, 1.0, 1.0, -0.5][:min(5, M)])
    s = (X.iloc[:, :len(w)].values @ w)
    if K == 2:
        y = (s > np.median(s)).astype(int)
    else:
        qs = np.quantile(s, np.linspace(0, 1, K + 1))
        y = np.digitize(s, qs[1:-1])
    return pd.Series(y, dtype="category")

def make_targets_discrete_from_real(X: pd.DataFrame, K: int = 3) -> pd.Series:
    """Generate classification labels from real features."""
    N, M = X.shape
    w = np.random.randn(M)
    s = X.values @ w + 0.25 * np.random.randn(N)
    qs = np.quantile(s, np.linspace(0, 1, K + 1))
    y = np.digitize(s, qs[1:-1])
    return pd.Series(y, dtype="category")

def make_targets_real_from_binary(X: pd.DataFrame) -> pd.Series:
    """Generate regression targets from binary features."""
    N, M = X.shape
    w = np.array([1.5, -2.0, 0.75, 1.0, -0.5][:min(5, M)])
    mu = (X.iloc[:, :len(w)].values @ w)
    y = mu + 0.5 * np.random.randn(N)
    return pd.Series(y.astype(float))

def make_targets_real_from_real(X: pd.DataFrame) -> pd.Series:
    """Generate regression targets from real features."""
    N, M = X.shape
    w = np.linspace(1, 2, M)
    y = (X.values @ w) + 0.5 * np.random.randn(N)
    return pd.Series(y.astype(float))


# -----------------------
# 2) TIMING UTILITIES
# -----------------------
def time_fit_predict(tree: DecisionTree, X_train: pd.DataFrame,
                     y_train: pd.Series, X_test: pd.DataFrame,
                     reps: int = 5) -> Tuple[float, float]:
    """Return average fit and predict time over `reps` runs."""
    fit_times, pred_times = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        tree.fit(X_train, y_train)
        t1 = time.perf_counter()

        _ = tree.predict(X_test)
        t2 = time.perf_counter()

        fit_times.append(t1 - t0)
        pred_times.append(t2 - t1)

    return float(np.mean(fit_times)), float(np.mean(pred_times))


def run_sweeps(N_list: List[int], M_list: List[int],
               reps: int = 5, max_depth: int = 5) -> Dict[str, dict]:
    """
    Run timing sweeps for all 4 cases:
      A) discrete features, discrete output
      B) discrete features, real output
      C) real features, discrete output
      D) real features, real output
    """
    results = {"disc-disc": {}, "disc-real": {}, "real-disc": {}, "real-real": {}}
    M_fixed = M_list[len(M_list)//2]
    N_fixed = N_list[len(N_list)//2]

    def _do_case(case_name: str, makeX, makeY, criterion: str):
        # --- vary N with M fixed ---
        fitN, predN = [], []
        for N in N_list:
            X, y = makeX(N, M_fixed), makeY(makeX(N, M_fixed))
            n_train = max(1, int(0.8 * N))
            Xtr, Xte, ytr, yte = X.iloc[:n_train], X.iloc[n_train:], y.iloc[:n_train], y.iloc[n_train:]
            tree = DecisionTree(criterion=criterion, max_depth=max_depth)
            f, p = time_fit_predict(tree, Xtr, ytr, Xte, reps=reps)
            fitN.append(f); predN.append(p)

        # --- vary M with N fixed ---
        fitM, predM = [], []
        for M in M_list:
            X, y = makeX(N_fixed, M), makeY(makeX(N_fixed, M))
            n_train = max(1, int(0.8 * N_fixed))
            Xtr, Xte, ytr, yte = X.iloc[:n_train], X.iloc[n_train:], y.iloc[:n_train], y.iloc[n_train:]
            tree = DecisionTree(criterion=criterion, max_depth=max_depth)
            f, p = time_fit_predict(tree, Xtr, ytr, Xte, reps=reps)
            fitM.append(f); predM.append(p)

        results[case_name]["vary_N"] = {"N": N_list, "fit": fitN, "pred": predN, "M_fixed": M_fixed}
        results[case_name]["vary_M"] = {"M": M_list, "fit": fitM, "pred": predM, "N_fixed": N_fixed}

    # A) Discrete → Discrete
    _do_case("disc-disc",
             makeX=lambda N, M: make_data_discrete_features(N, M),
             makeY=lambda X: make_targets_discrete_from_binary(X, K=2),
             criterion="information_gain")

    # B) Discrete → Real
    _do_case("disc-real",
             makeX=lambda N, M: make_data_discrete_features(N, M),
             makeY=make_targets_real_from_binary,
             criterion="information_gain")

    # C) Real → Discrete
    _do_case("real-disc",
             makeX=lambda N, M: make_data_real_features(N, M),
             makeY=lambda X: make_targets_discrete_from_real(X, K=3),
             criterion="gini_index")

    # D) Real → Real
    _do_case("real-real",
             makeX=lambda N, M: make_data_real_features(N, M),
             makeY=make_targets_real_from_real,
             criterion="information_gain")

    return results


# -----------------------
# 3) PLOTTING
# -----------------------
def plot_times(results: Dict[str, dict], save_prefix: str = None):
    """Plot fit & predict times vs N and M for each case."""
    for case, bundle in results.items():
        # vary N
        N, fit, pred, M_fixed = bundle["vary_N"]["N"], bundle["vary_N"]["fit"], bundle["vary_N"]["pred"], bundle["vary_N"]["M_fixed"]
        plt.figure()
        plt.plot(N, fit, marker="o", label="fit() time")
        plt.plot(N, pred, marker="s", label="predict() time")
        plt.xlabel("N (samples)")
        plt.ylabel("time (seconds)")
        plt.title(f"{case} | Vary N (M={M_fixed})")
        plt.legend(); plt.grid(True)
        if save_prefix:
            plt.savefig(f"{save_prefix}_{case}_varyN.png", bbox_inches="tight", dpi=160)
        plt.show()

        # vary M
        M, fit, pred, N_fixed = bundle["vary_M"]["M"], bundle["vary_M"]["fit"], bundle["vary_M"]["pred"], bundle["vary_M"]["N_fixed"]
        plt.figure()
        plt.plot(M, fit, marker="o", label="fit() time")
        plt.plot(M, pred, marker="s", label="predict() time")
        plt.xlabel("M (features)")
        plt.ylabel("time (seconds)")
        plt.title(f"{case} | Vary M (N={N_fixed})")
        plt.legend(); plt.grid(True)
        if save_prefix:
            plt.savefig(f"{save_prefix}_{case}_varyM.png", bbox_inches="tight", dpi=160)
        plt.show()


# -----------------------
# 4) THEORY QUICK REF
# -----------------------
def print_theory_cheatsheet():
    """Print standard time complexity formulas for decision trees."""
    txt = """
== Theoretical Time Complexity (coarse) ==
Let N = samples, M = features, d = depth, and T = #nodes (~O(2^d) worst-case).

Training (per node):
  • Discrete features: O(MN)
  • Numeric features: O(M N log N)

Overall training:
  • O(T * M N) (discrete)
  • O(T * M N log N) (numeric)

Prediction:
  • O(d) per sample → O(N_test * d)

Empirical expectation:
  • fit() vs N: roughly linear (steeper for numeric features due to log N).
  • fit() vs M: roughly linear.
  • predict() vs N_test: linear; vs M: weak dependence.
"""
    print(txt)


if __name__ == "__main__":
    # FAST preset for Colab
    N_list = [200, 400, 800]   # was [200, 400, 800, 1600]
    M_list = [4, 8, 16]        # was [4, 8, 16, 32]
    reps = 2                   # was 5
    max_depth = 4              # was 6

    print_theory_cheatsheet()

    # Small wrapper so we can pass one_hot_categoricals=False everywhere
    def run_sweeps_fast(N_list, M_list, reps, max_depth):
        results = {
            "disc-disc": {}, "disc-real": {}, "real-disc": {}, "real-real": {}
        }
        M_fixed = M_list[len(M_list)//2]
        N_fixed = N_list[len(N_list)//2]

        def _do_case(tag, makeX, makeY, criterion):
            print(f"\n== Running case: {tag} ==")
            # vary N
            fitN, predN = [], []
            for N in N_list:
                print(f"  vary N: N={N}, M={M_fixed}", flush=True)
                X = makeX(N, M_fixed); y = makeY(X)
                ntr = max(1, int(0.8*N))
                Xtr, Xte = X.iloc[:ntr], X.iloc[ntr:]; ytr, yte = y.iloc[:ntr], y.iloc[ntr:]
                tree = DecisionTree(criterion=criterion, max_depth=max_depth, one_hot_categoricals=False)
                f, p = time_fit_predict(tree, Xtr, ytr, Xte, reps=reps)
                fitN.append(f); predN.append(p)

            # vary M
            fitM, predM = [], []
            for M in M_list:
                print(f"  vary M: N={N_fixed}, M={M}", flush=True)
                X = makeX(N_fixed, M); y = makeY(X)
                ntr = max(1, int(0.8*N_fixed))
                Xtr, Xte = X.iloc[:ntr], X.iloc[ntr:]; ytr, yte = y.iloc[:ntr], y.iloc[ntr:]
                tree = DecisionTree(criterion=criterion, max_depth=max_depth, one_hot_categoricals=False)
                f, p = time_fit_predict(tree, Xtr, ytr, Xte, reps=reps)
                fitM.append(f); predM.append(p)

            results[tag]["vary_N"] = {"N": N_list, "fit": fitN, "pred": predN, "M_fixed": M_fixed}
            results[tag]["vary_M"] = {"M": M_list, "fit": fitM, "pred": predM, "N_fixed": N_fixed}

        # A) discrete features, discrete output
        _do_case("disc-disc",
                 makeX=lambda N, M: make_data_discrete_features(N, M),
                 makeY=lambda X: make_targets_discrete_from_binary(X, K=2),
                 criterion="information_gain")

        # B) discrete features, real output
        _do_case("disc-real",
                 makeX=lambda N, M: make_data_discrete_features(N, M),
                 makeY=make_targets_real_from_binary,
                 criterion="information_gain")  # ignored in regression

        # C) real features, discrete output
        _do_case("real-disc",
                 makeX=lambda N, M: make_data_real_features(N, M),
                 makeY=lambda X: make_targets_discrete_from_real(X, K=3),
                 criterion="gini_index")

        # D) real features, real output
        _do_case("real-real",
                 makeX=lambda N, M: make_data_real_features(N, M),
                 makeY=make_targets_real_from_real,
                 criterion="information_gain")  # ignored in regression

        return results

    results = run_sweeps_fast(N_list, M_list, reps=reps, max_depth=max_depth)
    plot_times(results, save_prefix="dt_runtime_fast")
    print("Done.")
