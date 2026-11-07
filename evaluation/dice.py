"""
evaluation/dice.py
------------------
Binary Dice Similarity Coefficient (DSC).
"""

import numpy as np


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Computes binary Dice coefficient between prediction and ground truth.

    Returns 1.0 if both prediction and ground truth are empty (sum == 0),
    representing perfect agreement on negative/background cases.
    """
    p = pred.astype(bool)
    g = gt.astype(bool)
    intersection = (p & g).sum()
    total = p.sum() + g.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


if __name__ == "__main__":
    a = np.array([1, 0, 1, 0])
    b = np.array([1, 1, 0, 0])
    assert dice(a, b) == 0.5
    assert dice(np.zeros(5), np.zeros(5)) == 1.0
    assert dice(np.ones(5), np.zeros(5)) == 0.0

