"""Apollon: per-cluster Multi-Armed Bandit with Thompson sampling over a pool of IDS classifiers.

How a request is answered:

1. Inputs are grouped with K-Means and every arm (classifier) is trained on each cluster,
   so each cluster has its own specialised pool.
2. Each (cluster, arm) pair keeps a Beta posterior over the arm's probability of
   classifying a request correctly. The posteriors start from the arms' performance on a
   held-out validation split.
3. For every request, Thompson sampling draws one value per arm from its posterior and the
   arm with the highest draw answers. Better arms answer more often, but the answering
   model keeps changing from request to request, so an attacker cannot learn a single,
   stable decision boundary to craft adversarial examples against.
4. When ground truth becomes available (for instance, analyst feedback), :meth:`update`
   adds the observed successes and failures to the posteriors, so the selection keeps
   adapting to the traffic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from apollon.pool import default_arms

__all__ = ["ApollonMAB", "load_model", "save_model"]


class ApollonMAB(ClassifierMixin, BaseEstimator):
    """Per-cluster Thompson-sampling selection over a pool of binary classifiers.

    The estimator follows the scikit-learn API (``fit``, ``predict``, ``predict_proba``,
    ``get_params``), so it can be used wherever a classifier is expected.

    Parameters
    ----------
    arms:
        Unfitted scikit-learn classifiers. ``None`` uses the pool of the paper
        (:func:`apollon.pool.default_arms`).
    n_clusters:
        Number of K-Means clusters (the paper uses 2).
    val_size:
        Fraction of the training data held out to initialise the Beta posteriors.
    prior_strength:
        If set, the validation counts of each (cluster, arm) are rescaled to this many
        pseudo-observations, which keeps the selection more mixed. ``None`` uses the raw
        counts (plain Thompson sampling).
    random_state:
        Seed for the split, K-Means, the arms and the Thompson draws.

    Attributes
    ----------
    kmeans_:
        Fitted K-Means model.
    models_:
        ``models_[c][a]`` is arm ``a`` trained on cluster ``c``.
    alpha_, beta_:
        Beta posterior parameters, shape ``(n_clusters, n_arms)``.
    shared_:
        ``True`` where a cluster contained a single class and the arm trained on all the
        data is used for it.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from apollon import ApollonMAB
    >>> X, y = make_classification(n_samples=500, random_state=0)
    >>> model = ApollonMAB(random_state=0).fit(X, y)
    >>> labels, arms = model.predict(X[:5], return_arms=True)
    """

    def __init__(
        self,
        arms=None,
        n_clusters: int = 2,
        val_size: float = 0.2,
        prior_strength: float | None = None,
        random_state: int | None = None,
    ):
        self.arms = arms
        self.n_clusters = n_clusters
        self.val_size = val_size
        self.prior_strength = prior_strength
        self.random_state = random_state

    # ------------------------------------------------------------------ training

    @property
    def n_arms(self) -> int:
        return len(self._arms())

    def _arms(self):
        return self.arms if self.arms is not None else default_arms(self.random_state)

    def fit(self, X, y):
        """Cluster the data, train every arm per cluster and initialise the posteriors."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        self.rng_ = np.random.default_rng(self.random_state)
        arms = self._arms()
        X_fit, X_val, y_fit, y_val = train_test_split(
            X, y, test_size=self.val_size, random_state=self.random_state, stratify=y
        )

        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters, n_init=10, random_state=self.random_state
        ).fit(X_fit)
        c_fit = self.kmeans_.labels_
        c_val = self.kmeans_.predict(X_val)

        self.models_ = []
        self.shared_ = np.zeros((self.n_clusters, len(arms)), dtype=bool)
        global_models = None
        for c in range(self.n_clusters):
            m = c_fit == c
            row = []
            for a, arm in enumerate(arms):
                if len(np.unique(y_fit[m])) > 1:
                    row.append(clone(arm).fit(X_fit[m], y_fit[m]))
                else:
                    # A cluster with a single class cannot train a classifier on its own:
                    # use the arm trained on all the data for that cluster.
                    if global_models is None:
                        global_models = [clone(a_).fit(X_fit, y_fit) for a_ in arms]
                    row.append(global_models[a])
                    self.shared_[c, a] = True
            self.models_.append(row)

        self.alpha_ = np.ones((self.n_clusters, len(arms)))
        self.beta_ = np.ones((self.n_clusters, len(arms)))
        for c in range(self.n_clusters):
            m = c_val == c
            n = int(m.sum())
            if n == 0:
                continue
            for a in range(len(arms)):
                correct = float(np.sum(self.models_[c][a].predict(X_val[m]) == y_val[m]))
                scale = 1.0 if self.prior_strength is None else self.prior_strength / n
                self.alpha_[c, a] += correct * scale
                self.beta_[c, a] += (n - correct) * scale
        return self

    # ------------------------------------------------------------------ selection

    def assign_clusters(self, X):
        """K-Means cluster of every row of ``X``."""
        return self.kmeans_.predict(np.asarray(X, dtype=float))

    def select_arms(self, X):
        """Cluster and Thompson-selected arm for every row of ``X``."""
        clusters = self.assign_clusters(X)
        theta = self.rng_.beta(self.alpha_[clusters], self.beta_[clusters])
        return clusters, np.argmax(theta, axis=1)

    def _route(self, X, method):
        X = np.asarray(X, dtype=float)
        clusters, arms = self.select_arms(X)
        out = None
        for c in range(self.n_clusters):
            for a in range(self.alpha_.shape[1]):
                m = (clusters == c) & (arms == a)
                if not m.any():
                    continue
                res = getattr(self.models_[c][a], method)(X[m])
                if out is None:
                    out = np.zeros((len(X), *res.shape[1:]), dtype=float)
                out[m] = res
        return out, clusters, arms

    def predict(self, X, return_arms: bool = False):
        """Predicted label of every row (one Thompson draw per row).

        With ``return_arms=True`` also returns the index of the arm that answered.
        """
        y, _, arms = self._route(X, "predict")
        y = y.astype(int)
        return (y, arms) if return_arms else y

    def predict_proba(self, X):
        """Class probabilities from the arm selected for each row (one draw per row)."""
        return self._route(X, "predict_proba")[0]

    # ------------------------------------------------------------------ feedback

    def update(self, X, y_true, clusters=None, arms=None, y_pred=None):
        """Update the posteriors with ground truth for requests already answered.

        Pass the ``clusters``/``arms`` returned when the requests were answered (and the
        predictions) to credit exactly the arms that answered. Without them, the requests
        are answered again now.
        """
        y_true = np.asarray(y_true).astype(int)
        if clusters is None or arms is None or y_pred is None:
            y_pred, arms = self.predict(X, return_arms=True)
            clusters = self.assign_clusters(X)
        correct = (np.asarray(y_pred) == y_true).astype(float)
        np.add.at(self.alpha_, (clusters, arms), correct)
        np.add.at(self.beta_, (clusters, arms), 1.0 - correct)
        return self

    # ------------------------------------------------------------------ inspection

    def selection_frequencies(self, X, n_draws: int = 1) -> np.ndarray:
        """Share of requests answered by each arm."""
        counts = np.zeros(self.alpha_.shape[1])
        for _ in range(n_draws):
            counts += np.bincount(self.select_arms(X)[1], minlength=self.alpha_.shape[1])
        return counts / counts.sum()

    def posterior_means(self) -> np.ndarray:
        """Expected accuracy of every (cluster, arm), shape ``(n_clusters, n_arms)``."""
        return self.alpha_ / (self.alpha_ + self.beta_)


def save_model(model: ApollonMAB, path: str | Path) -> Path:
    """Save a fitted model with joblib."""
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path) -> ApollonMAB:
    """Load a model saved with :func:`save_model`.

    Only load files you trust: joblib files can execute code when loaded.
    """
    import joblib

    return joblib.load(path)
