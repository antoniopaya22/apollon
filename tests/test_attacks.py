"""Smoke tests for attacks/attack_utils.py on synthetic data (small budgets)."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from attacks.attack_utils import (
    FUNCTIONAL_FEATURES,
    apollon_classifier,
    evaluate,
    functional_indexes,
    model_classifier,
    restore_functional,
    run_attack,
)
from MAB.apollon_mab import ApollonMAB


def test_functional_feature_list_is_complete():
    assert len(FUNCTIONAL_FEATURES) == 27
    assert len(set(FUNCTIONAL_FEATURES)) == 27


def test_functional_indexes_and_restore():
    cols = [" Destination Port", " Flow Duration", " Total Fwd Packets", "Other"]
    func, mod, missing = functional_indexes(cols)
    assert func == [0, 2] and mod == [1, 3]
    assert " Destination Port" not in missing and len(missing) == 25
    x = np.zeros((3, 4))
    adv = np.ones((3, 4))
    out = restore_functional(adv, x, func)
    assert np.all(out[:, func] == 0) and np.all(out[:, mod] == 1)


def _data():
    X, y = make_classification(n_samples=1500, n_features=8, n_informative=5,
                               weights=[0.6, 0.4], random_state=1)
    return MinMaxScaler().fit_transform(X), y


def test_direct_attacks_query_apollon():
    X, y = _data()
    mab = ApollonMAB(n_clusters=2, random_state=0).fit(X[:1200], y[:1200])
    clf = apollon_classifier(mab, X.shape[1])
    calls = {"n": 0}
    orig = mab.predict_proba

    def counted(x):
        calls["n"] += len(x)
        return orig(x)

    mab.predict_proba = counted
    x_atk = X[1200:][y[1200:] == 1][:5]
    for name in ("hsja", "zoo"):
        before = calls["n"]
        adv = run_attack(name, clf, x_atk, max_iter=2, nb_parallel=4)
        assert adv.shape == x_atk.shape
        assert calls["n"] > before  # the attack really queried the full system


def test_transfer_attack_and_single_prediction_metrics():
    X, y = _data()
    dt = DecisionTreeClassifier(random_state=0).fit(X[:1200], y[:1200])
    x_atk = X[1200:][y[1200:] == 1][:5]
    adv = run_attack("hsja", model_classifier(dt), x_atk, max_iter=2)
    m = evaluate(np.ones(len(adv)).tolist() + [0] * 5,
                 dt.predict(np.vstack([adv, X[1200:][y[1200:] == 0][:5]])))
    assert set(m) == {"accuracy", "detection_rate", "f1", "roc_auc"}
