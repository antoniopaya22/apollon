"""Attack helpers used by the ZOO and HopSkipJump notebooks of this folder.

The implementation lives in the installable package (``src/apollon/attacks.py``); install it
from the repository root with ``pip install -e ".[attacks]"``.
"""

from apollon.attacks import (  # noqa: F401
    FUNCTIONAL_FEATURES,
    apollon_classifier,
    functional_indexes,
    model_classifier,
    restore_functional,
    run_attack,
)
from apollon.metrics import evaluate as _evaluate


def evaluate(y_true, y_pred):
    """Accuracy, detection rate, F1 and ROC AUC of a single prediction."""
    m = _evaluate(y_true, y_pred)
    return {k: m[k] for k in ("accuracy", "detection_rate", "f1", "roc_auc")}
