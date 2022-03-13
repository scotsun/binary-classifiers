"""Several simple classifiers."""
import matplotlib.pyplot as plt
import numpy as np


def lp_distance(pt1: np.ndarray, pt2: np.ndarray, p: int | float) -> np.ndarray:
    """Calculate the p-norm distance."""
    return (((pt1 - pt2) ** p).sum(axis=1)) ** (1 / p)


class KNN:
    """KNN classifier."""

    def __init__(self, k: int, p: int | float) -> None:
        """Initialize."""
        self._k = k
        self._p = p

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the data."""
        self._X = X
        self._y = y

    def _get_neighbors_idx(self, test_x: np.ndarray) -> np.ndarray:
        try:
            dists = lp_distance(self._X, test_x, self._p)
        except AttributeError as e:
            raise AttributeError("the model hasn't fit any data yet")
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

    @property
    def training_score(self) -> float | None:
        """Training accuracy."""
        try:
            score = (self.predict(self._X) == self._y).mean()
            return score
        except AttributeError as e:
            raise AttributeError("the model hasn't fit any data yet")

    def predict(self, test: np.ndarray) -> np.ndarray:
        """Predict for the entire test set."""
        return np.array([self._make_classification(test_row) for test_row in test])

    def plot_decision_boundary(
        self, x1_lims: tuple[float, float], x2_lims: tuple[float, float], step=5e-2
    ) -> None:
        """Plot the decision boundary based on the training data."""
        x1 = np.arange(x1_lims[0], x1_lims[1], step)
        x2 = np.arange(x2_lims[0], x2_lims[1], step)
        X1, X2 = np.meshgrid(x1, x2)
        points = np.concatenate((X1.reshape(-1, 1), X2.reshape(-1, 1)), axis=1)
        points_labels = self.predict(points)
        points_labels = points_labels.reshape(X1.shape)
        plt.figure()
        plt.pcolormesh(X1, X2, points_labels, alpha=0.3)
        plt.scatter(self._X[:, 0], self._X[:, 1], c=self._y)
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        plt.title(f"KNN with {self._k} neighbors {self._p} norm")
        plt.show()


# TODO:
#   LDA:
#       find hyperplane wTx, s.t. w maximize diff_mean^2 / sum(group_var[i])
#
#   Naive Bayes ML:
#       find argmax_y for p(y|x) ~ p(x|y) * p(y)


if __name__ == "__main__":
    n = 100
    mu_x1, mu_y1 = 0, 0
    sd_x1, sd_y1 = 1, 1

    mu_x2, mu_y2 = 2, 2
    sd_x2, sd_y2 = 1, 1

    x1 = np.random.normal(mu_x1, sd_x1, n)
    y1 = np.random.normal(mu_y1, sd_y1, n)

    x2 = np.random.normal(mu_x2, sd_x2, n)
    y2 = np.random.normal(mu_y2, sd_y2, n)

    x = np.concatenate([x1, x2], axis=0)
    y = np.concatenate([y1, y2], axis=0)
    dat = np.vstack((x, y)).T
    target = np.array([0] * n + [1] * n).reshape(2 * n, 1)
    total_data = np.hstack((dat, target))

    train_idx = np.random.randint(0, 2 * n, size=int(0.8 * 2 * n))
    train_total_dat = total_data[train_idx]
    test_total_dat = total_data[-train_idx]

    knn = KNN(10, 2)
    knn.fit(X=train_total_dat[:, :2], y=train_total_dat[:, 2])
    knn.plot_decision_boundary(x1_lims=(-4, 4), x2_lims=(-3, 5))
