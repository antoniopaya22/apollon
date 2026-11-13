"""HopSkipJump against Apollon (direct setting) and against a single Random Forest.

Needs the attacks extra: pip install "apollon-ids[attacks]".

    python examples/adversarial.py
"""

import numpy as np
from sklearn.datasets import make_classification

from apollon import ApollonMAB, make_arm
from apollon.attacks import apollon_classifier, model_classifier, run_attack
from apollon.data import split_and_scale

X, y = make_classification(
    n_samples=3000, n_features=15, n_informative=10, weights=[0.7, 0.3], random_state=0
)
data = split_and_scale(X, y, seed=0)
apollon = ApollonMAB(random_state=0).fit(data.X_train, data.y_train)
forest = make_arm("RF", 0).fit(data.X_train, data.y_train)

x = data.X_test[data.y_test == 1][:20]
for name, target, clf in [
    ("Random Forest", forest, model_classifier(forest)),
    ("Apollon", apollon, apollon_classifier(apollon, x.shape[1])),
]:
    adv = run_attack("hsja", clf, x, max_iter=10)
    print(
        f"{name:<14} detected before {np.mean(target.predict(x) == 1):.0%}, "
        f"after {np.mean(target.predict(adv) == 1):.0%}"
    )
