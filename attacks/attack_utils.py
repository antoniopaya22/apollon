"""Shared helpers for the ZOO and HopSkipJump notebooks.

Fixes with respect to the previous notebook code:

* ``FUNCTIONAL_FEATURES`` was missing commas, so Python concatenated adjacent strings and the
  list had 19 entries instead of 27; several names never matched a column.
* The indexes were computed on ``df.columns`` (raw CSV, before columns are dropped during
  preprocessing) and the names were swapped (``FUNCTIONAL_FEATURES_IDEXES`` held the
  *non*-functional columns). They are now computed on the columns actually fed to the models.
* The ``Classifier`` wrapper that should keep functional features unchanged was commented out
  (and returned only the first prediction). ``restore_functional`` enforces the constraint
  after the attack instead.
* Attacks were generated against single models and only *replayed* on Apollon, which measures
  transferability. ``apollon_classifier`` wraps the whole system so attacks can query it.
* Metrics were computed from several independent ``mab.predict`` calls; with random arm
  selection each call answers differently. ``evaluate`` uses a single prediction.
"""

from __future__ import annotations

import numpy as np
from art.attacks.evasion import HopSkipJump, ZooAttack
from art.estimators.classification import BlackBoxClassifier, SklearnClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

# Features that must keep their value for the attack to remain functional (CIC-IDS-2017
# column names, as in the original CSV headers).
FUNCTIONAL_FEATURES = [
    " min_seg_size_forward", " Bwd Header Length", " Destination Port",
    "Init_Win_bytes_forward", " Init_Win_bytes_backward", " Bwd Packets/s",
    "Total Length of Fwd Packets", " Subflow Fwd Bytes", " Max Packet Length",
    "Bwd Packet Length Max", " Avg Bwd Segment Size", " Bwd Packet Length Mean",
    " Fwd Packet Length Max", " Average Packet Size", " Packet Length Std",
    " Packet Length Mean", " Bwd Packet Length Std", " Bwd Packet Length Min",
    " Fwd Packet Length Std", " Fwd Packet Length Min", " Min Packet Length",
    " Fwd Packet Length Mean", " Avg Fwd Segment Size", " act_data_pkt_fwd",
    " Total Fwd Packets", "Subflow Fwd Packets", " Total Backward Packets",
]

BENIGN, ATTACK = 0, 1


def functional_indexes(columns):
    """Indexes of functional (must not change) and modifiable features in ``columns``.

    ``columns`` are the feature columns in the order the models receive them, e.g.
    ``df_preprocessed.x.columns``. Returns ``(functional, modifiable, missing)`` where
    ``missing`` lists functional names that are not among the columns.
    """
    columns = list(columns)
    functional = [i for i, c in enumerate(columns) if c in FUNCTIONAL_FEATURES]
    modifiable = [i for i, c in enumerate(columns) if c not in FUNCTIONAL_FEATURES]
    missing = [c for c in FUNCTIONAL_FEATURES if c not in columns]
    return functional, modifiable, missing


def restore_functional(x_adv, x_orig, functional_idx):
    """Put back the original value of every functional feature."""
    x_adv = np.array(x_adv, dtype=float, copy=True)
    x_adv[:, functional_idx] = np.asarray(x_orig, dtype=float)[:, functional_idx]
    return x_adv


def model_classifier(model, clip_values=(0, 1)):
    """ART wrapper around a single scikit-learn model (transfer setting)."""
    return SklearnClassifier(model=model, clip_values=clip_values)


def apollon_classifier(mab, n_features, clip_values=(0, 1)):
    """ART wrapper that queries the whole Apollon system.

    Every query goes through cluster assignment and Thompson-sampling arm selection, exactly
    as a request reaching the deployed IDS. Probabilities come from the arm selected for that
    query (ZOO needs scores; HopSkipJump only uses the predicted label).
    """
    return BlackBoxClassifier(predict_fn=lambda x: mab.predict_proba(x),
                              input_shape=(n_features,), nb_classes=2,
                              clip_values=clip_values)


def run_attack(name, classifier, x, max_iter=100, nb_parallel=60):
    """Targeted attack towards the benign class. ``max_iter`` follows the paper (100)."""
    y_target = np.zeros((len(x), 2))
    y_target[:, BENIGN] = 1
    if name == "zoo":
        attack = ZooAttack(classifier=classifier, targeted=True, max_iter=max_iter,
                           nb_parallel=nb_parallel, verbose=False)
    elif name == "hsja":
        # The previous call used targeted=False with label 0 for attack samples, i.e. it
        # asked for "anything but benign", which an already detected attack satisfies.
        attack = HopSkipJump(classifier=classifier, targeted=True, max_iter=max_iter,
                             verbose=False)
    else:
        raise ValueError(name)
    return attack.generate(x=np.asarray(x, dtype=float), y=y_target)


def evaluate(y_true, y_pred):
    """Metrics computed once from a single prediction (attack class = 1)."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "detection_rate": recall_score(y_true, y_pred, pos_label=ATTACK),
        "f1": f1_score(y_true, y_pred, pos_label=ATTACK),
        "roc_auc": roc_auc_score(y_true, y_pred),
    }
