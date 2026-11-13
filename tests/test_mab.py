"""Tests of the ApollonMAB estimator on synthetic data (no dataset needed)."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from apollon import ApollonMAB, load_model, save_model


def data(seed=0, n=3000):
    X, y = make_classification(
        n_samples=n, n_features=12, n_informative=6, weights=[0.7, 0.3], random_state=seed
    )
    return X, y


def test_fit_predict_shapes_and_accuracy():
    X, y = data()
    mab = ApollonMAB(n_clusters=2, random_state=0).fit(X[:2000], y[:2000])
    pred, arms = mab.predict(X[2000:], return_arms=True)
    assert pred.shape == (1000,) and arms.shape == (1000,)
    assert set(np.unique(pred)) <= {0, 1}
    assert (pred == y[2000:]).mean() > 0.8
    proba = mab.predict_proba(X[2000:])
    assert proba.shape == (1000, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_rewards_use_all_validation_samples_of_the_cluster():
    """Each arm is scored on every validation sample of its cluster."""
    X, y = data()
    arms = [
        DummyClassifier(strategy="constant", constant=0),
        DummyClassifier(strategy="constant", constant=1),
        RandomForestClassifier(n_estimators=50, random_state=0),
    ]
    mab = ApollonMAB(arms=arms, n_clusters=1, random_state=0).fit(X, y)
    mean = mab.alpha_ / (mab.alpha_ + mab.beta_)
    # Constant-0 is right on ~70% of the samples, constant-1 on ~30%, the forest on most.
    assert 0.6 < mean[0, 0] < 0.8
    assert 0.2 < mean[0, 1] < 0.4
    assert mean[0, 2] > 0.85


def test_arms_are_trained_per_cluster():
    X, y = data()
    mab = ApollonMAB(arms=[GaussianNB()], n_clusters=2, random_state=0).fit(X, y)
    m0, m1 = mab.models_[0][0], mab.models_[1][0]
    assert m0 is not m1
    assert not np.allclose(m0.theta_, m1.theta_)
    assert not mab.shared_.any()


def test_thompson_selection_prefers_better_arms():
    X, y = data()
    arms = [
        DummyClassifier(strategy="constant", constant=1),
        RandomForestClassifier(n_estimators=50, random_state=0),
    ]
    mab = ApollonMAB(arms=arms, n_clusters=1, random_state=0).fit(X, y)
    freq = mab.selection_frequencies(X, n_draws=3)
    assert freq[1] > 0.95


def test_prior_strength_keeps_selection_mixed():
    X, y = data()
    arms = [GaussianNB(), RandomForestClassifier(n_estimators=50, random_state=0)]
    sharp = ApollonMAB(arms=arms, n_clusters=1, random_state=0).fit(X, y)
    soft = ApollonMAB(arms=arms, n_clusters=1, prior_strength=2, random_state=0).fit(X, y)
    assert soft.selection_frequencies(X)[0] > sharp.selection_frequencies(X)[0]


def test_update_learns_from_feedback():
    X, y = data()
    arms = [
        DummyClassifier(strategy="constant", constant=0),
        DummyClassifier(strategy="constant", constant=1),
    ]
    mab = ApollonMAB(arms=arms, n_clusters=1, random_state=0).fit(X, y)
    before = mab.selection_frequencies(X)[1]
    # Feedback stream made only of attacks: the constant-1 arm is always right.
    Xa = X[y == 1]
    for _ in range(5):
        pred, arms_used = mab.predict(Xa, return_arms=True)
        mab.update(
            Xa, np.ones(len(Xa)), clusters=np.zeros(len(Xa), dtype=int), arms=arms_used, y_pred=pred
        )
    assert mab.selection_frequencies(X)[1] > before


@pytest.mark.parametrize("labels", [["0", "1"], [0, 1]])
def test_string_labels_are_accepted(labels):
    X, y = data(n=600)
    y = np.array([labels[v] for v in y])
    ApollonMAB(n_clusters=2, random_state=0).fit(X, y).predict(X[:10])


def test_sklearn_api_and_persistence(tmp_path):
    from sklearn.base import clone

    X, y = data(n=800)
    mab = ApollonMAB(n_clusters=2, random_state=0).fit(X, y)
    assert clone(mab).get_params()["n_clusters"] == 2
    assert list(mab.classes_) == [0, 1]
    assert mab.posterior_means().shape == (2, mab.n_arms)
    path = save_model(mab, tmp_path / "m.joblib")
    loaded = load_model(path)
    np.testing.assert_array_equal(loaded.alpha_, mab.alpha_)
    assert loaded.predict(X[:20]).shape == (20,)


def test_same_seed_same_answers():
    X, y = data(n=800)
    a = ApollonMAB(random_state=3).fit(X, y).predict(X[:200])
    b = ApollonMAB(random_state=3).fit(X, y).predict(X[:200])
    np.testing.assert_array_equal(a, b)


def test_float32_input_and_cluster_assignment():
    X, y = data(n=800)
    mab = ApollonMAB(n_clusters=2, random_state=0).fit(X, y)
    X32 = X[:50].astype(np.float32)
    clusters = mab.assign_clusters(X32)
    pred, arms = mab.predict(X32, return_arms=True)
    mab.update(X32, y[:50], clusters=clusters, arms=arms, y_pred=pred)
    assert set(np.unique(clusters)) <= {0, 1}
