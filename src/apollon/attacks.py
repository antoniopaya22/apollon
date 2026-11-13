"""Adversarial evaluation with the Adversarial Robustness Toolbox (ART).

Two settings are supported:

* **transfer**: adversarial examples are crafted against a single classifier
  (:func:`model_classifier`) and then submitted to Apollon;
* **direct**: the attack queries Apollon itself (:func:`apollon_classifier`), so every
  query goes through cluster assignment and Thompson-sampled selection, exactly as a
  request reaching the deployed IDS.

Both attacks are black-box and targeted towards the benign class: ZOO (score-based) and
HopSkipJump (decision-based). After an attack, :func:`restore_functional` puts back the
value of every *functional* feature, those an attacker cannot change without breaking
the attack itself.

Requires the ``attacks`` extra: ``pip install "apollon-ids[attacks]"``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ATTACKS",
    "FUNCTIONAL_FEATURES",
    "apollon_classifier",
    "functional_indexes",
    "model_classifier",
    "restore_functional",
    "run_attack",
]

#: Supported attacks.
ATTACKS = ("zoo", "hsja")

BENIGN, ATTACK = 0, 1

#: Features that must keep their value for an attack to remain functional (CIC-IDS2017
#: column names, as in the original CSV headers, leading spaces included).
FUNCTIONAL_FEATURES = [
    " min_seg_size_forward",
    " Bwd Header Length",
    " Destination Port",
    "Init_Win_bytes_forward",
    " Init_Win_bytes_backward",
    " Bwd Packets/s",
    "Total Length of Fwd Packets",
    " Subflow Fwd Bytes",
    " Max Packet Length",
    "Bwd Packet Length Max",
    " Avg Bwd Segment Size",
    " Bwd Packet Length Mean",
    " Fwd Packet Length Max",
    " Average Packet Size",
    " Packet Length Std",
    " Packet Length Mean",
    " Bwd Packet Length Std",
    " Bwd Packet Length Min",
    " Fwd Packet Length Std",
    " Fwd Packet Length Min",
    " Min Packet Length",
    " Fwd Packet Length Mean",
    " Avg Fwd Segment Size",
    " act_data_pkt_fwd",
    " Total Fwd Packets",
    "Subflow Fwd Packets",
    " Total Backward Packets",
]


def _art():
    try:
        import art  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError('adversarial attacks need ART: pip install "apollon-ids[attacks]"') from e


def functional_indexes(columns, functional=FUNCTIONAL_FEATURES):
    """Indexes of functional (must not change) and modifiable features in ``columns``.

    ``columns`` are the feature columns in the order the models receive them. Returns
    ``(functional, modifiable, missing)``, where ``missing`` lists functional names that
    are not among the columns.
    """
    columns = [str(c) for c in columns]
    names = set(functional)
    func = [i for i, c in enumerate(columns) if c in names]
    mod = [i for i, c in enumerate(columns) if c not in names]
    missing = [c for c in functional if c not in columns]
    return func, mod, missing


def restore_functional(x_adv, x_orig, functional_idx):
    """Put back the original value of every functional feature."""
    x_adv = np.array(x_adv, dtype=float, copy=True)
    x_adv[:, functional_idx] = np.asarray(x_orig, dtype=float)[:, functional_idx]
    return x_adv


def model_classifier(model, clip_values=(0, 1)):
    """ART wrapper around a single scikit-learn model (transfer setting)."""
    _art()
    from art.estimators.classification import SklearnClassifier

    return SklearnClassifier(model=model, clip_values=clip_values)


def apollon_classifier(model, n_features: int, clip_values=(0, 1)):
    """ART wrapper that queries the whole Apollon system (direct setting).

    Probabilities come from the arm selected for each query: ZOO uses the scores and
    HopSkipJump only the predicted label.
    """
    _art()
    from art.estimators.classification import BlackBoxClassifier

    return BlackBoxClassifier(
        predict_fn=lambda x: model.predict_proba(x),
        input_shape=(n_features,),
        nb_classes=2,
        clip_values=clip_values,
    )


def run_attack(name: str, classifier, x, max_iter: int = 100, nb_parallel: int = 60):
    """Run a targeted (towards benign) black-box attack and return the adversarial examples.

    Parameters
    ----------
    name:
        ``"zoo"`` or ``"hsja"`` (HopSkipJump).
    classifier:
        An ART classifier, from :func:`model_classifier` or :func:`apollon_classifier`.
    x:
        Attack samples to perturb.
    max_iter:
        Iterations of the attack (100 in the paper).
    nb_parallel:
        Coordinates updated in parallel by ZOO.
    """
    _art()
    from art.attacks.evasion import HopSkipJump, ZooAttack

    y_target = np.zeros((len(x), 2))
    y_target[:, BENIGN] = 1
    if name == "zoo":
        attack = ZooAttack(
            classifier=classifier,
            targeted=True,
            max_iter=max_iter,
            nb_parallel=nb_parallel,
            verbose=False,
        )
    elif name == "hsja":
        attack = HopSkipJump(classifier=classifier, targeted=True, max_iter=max_iter, verbose=False)
    else:
        raise ValueError(f"unknown attack {name!r}; available: {', '.join(ATTACKS)}")
    return attack.generate(x=np.asarray(x, dtype=float), y=y_target)
