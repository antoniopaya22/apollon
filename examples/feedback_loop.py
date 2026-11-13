"""Online learning: analysts label a batch of answered requests and Apollon updates its
posteriors, so the arms that were right answer more often.

    python examples/feedback_loop.py
"""

import numpy as np
from sklearn.datasets import make_classification

from apollon import ApollonMAB
from apollon.data import split_and_scale

X, y = make_classification(
    n_samples=6000, n_features=20, n_informative=12, weights=[0.8, 0.2], random_state=1
)
data = split_and_scale(X, y, seed=1, normalize=False)
model = ApollonMAB(prior_strength=10, random_state=1).fit(data.X_train, data.y_train)

stream_X, stream_y = data.X_test, data.y_test
for i, batch in enumerate(np.array_split(np.arange(len(stream_y)), 6)):
    Xb, yb = stream_X[batch], stream_y[batch]
    pred, arms = model.predict(Xb, return_arms=True)
    clusters = model.assign_clusters(Xb)
    print(
        f"batch {i}: accuracy {np.mean(pred == yb):.3f}, "
        f"arm mix {np.round(np.bincount(arms, minlength=model.n_arms) / len(arms), 2)}"
    )
    model.update(Xb, yb, clusters=clusters, arms=arms, y_pred=pred)  # analyst feedback
