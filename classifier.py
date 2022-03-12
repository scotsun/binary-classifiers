"""Several simple classifiers."""
import numpy as np


def lp_distance(pt1: np.ndarray, pt2: np.ndarray, p: int | float) -> np.ndarray:
    """Calculate the p-norm distance."""
    return (((pt1 - pt2) ** p).sum(axis=1)) ** (1 / p)


class KNN:
    """KNN classifier."""

    def __init__(self, X: np.ndarray, y: np.ndarray, k: int, p: int | float) -> None:
        """Initialize."""
        self._X = X
        self._y = y
        self._k = k
        self._p = p

    def _get_neighbors_idx(self, test_x: np.ndarray) -> np.ndarray:
        dists = lp_distance(self._X, test_x, self._p)
        indexed_dists = np.vstack([range(len(dists)), dists])
        indexed_dists = indexed_dists[:, np.argsort(indexed_dists[1, :])]
        neighbors = indexed_dists[0, : self._k]
        return neighbors

    def _make_classification(self, test_x: np.ndarray) -> np.intp:
        """Point estimation."""
        neighbors_idx = self._get_neighbors_idx(test_x).astype("int")
        neighbors_y = self._y[neighbors_idx].astype("int")
        test_yh = np.bincount(neighbors_y).argmax()
        return test_yh

    def predict(self, test: np.ndarray) -> np.ndarray:
        """Predict for the entire test set."""
        return np.array([self._make_classification(test_row) for test_row in test])


# TODO:
#   LDA:
#       find hyperplane wTx, s.t. w maximize diff_mean^2 / sum(group_var[i])
#
#   Naive Bayes ML:
#       find argmax_y for p(y|x) ~ p(x|y) * p(y)
