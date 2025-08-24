from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_cap: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy.
    """
    assert y_cap.size == y.size, "Input series must have the same size."
    return (y_cap == y).sum() / len(y)

def precision(y_cap: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision.
    """
    assert y_cap.size == y.size, "Input series must have the same size."
    true_positives = ((y_cap == cls) & (y == cls)).sum()
    predicted_positives = (y_cap == cls).sum()
    return true_positives / predicted_positives if predicted_positives > 0 else 0.0

def recall(y_cap: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall.
    """
    assert y_cap.size == y.size, "Input series must have the same size."
    true_positives = ((y_cap == cls) & (y == cls)).sum()
    actual_positives = (y == cls).sum()
    return true_positives / actual_positives if actual_positives > 0 else 0.0

def rmse(y_cap: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse).
    """
    assert y_cap.size == y.size, "Input series must have the same size."
    return np.sqrt(np.mean((y_cap - y)**2))

def mae(y_cap: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae).
    """
    assert y_cap.size == y.size, "Input series must have the same size."
    return np.mean(np.abs(y_cap - y))
