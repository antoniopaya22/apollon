"""The pool of intrusion detection classifiers that Apollon selects from.

Every *arm* of the bandit is a scikit-learn classifier. The default pool is the one used in
the paper: Random Forest, Decision Tree, Gaussian Naive Bayes, Logistic Regression and a
Multi-Layer Perceptron. Any other classifier with ``fit``/``predict``/``predict_proba``
can be added.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

__all__ = ["ARMS", "DEFAULT_POOL", "default_arms", "make_arm", "make_pool"]

ArmFactory = Callable[[int | None], ClassifierMixin]

#: Registry of the available arms, by short name.
ARMS: dict[str, ArmFactory] = {
    "RF": lambda seed: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed),
    "DT": lambda seed: DecisionTreeClassifier(random_state=seed),
    "NB": lambda seed: GaussianNB(),
    "LR": lambda seed: LogisticRegression(max_iter=1000),
    "MLP": lambda seed: MLPClassifier(
        hidden_layer_sizes=(32,),
        max_iter=200,
        early_stopping=True,
        activation="tanh",
        solver="adam",
        random_state=seed,
    ),
    "KNN": lambda seed: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "SVC": lambda seed: SVC(probability=True, random_state=seed),
}

#: Arms used in the paper, in order.
DEFAULT_POOL: tuple[str, ...] = ("RF", "DT", "NB", "LR", "MLP")


def make_arm(name: str, random_state: int | None = None) -> ClassifierMixin:
    """Build one unfitted arm from its short name (see :data:`ARMS`)."""
    try:
        return ARMS[name.upper()](random_state)
    except KeyError:
        raise ValueError(f"unknown arm {name!r}; available: {', '.join(ARMS)}") from None


def make_pool(
    names: Iterable[str] = DEFAULT_POOL, random_state: int | None = None
) -> list[ClassifierMixin]:
    """Build a pool of unfitted arms from their short names."""
    return [make_arm(n, random_state) for n in names]


def default_arms(random_state: int | None = None) -> list[ClassifierMixin]:
    """The classifier pool of the paper: RF, DT, NB, LR and MLP."""
    return make_pool(DEFAULT_POOL, random_state)
