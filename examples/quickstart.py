"""Train Apollon on synthetic traffic and inspect the arm mix.

python examples/quickstart.py
"""

from sklearn.datasets import make_classification

from apollon import DEFAULT_POOL, ApollonMAB
from apollon.data import split_and_scale
from apollon.metrics import evaluate, format_metrics

X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=12,
    n_redundant=4,
    class_sep=1.5,
    weights=[0.8, 0.2],
    random_state=0,
)
data = split_and_scale(X, y, seed=0, normalize=False)

model = ApollonMAB(n_clusters=2, random_state=0).fit(data.X_train, data.y_train)
pred = model.predict(data.X_test)
print(format_metrics(evaluate(data.y_test, pred)))

print("\nExpected accuracy per cluster and arm:")
for c, row in enumerate(model.posterior_means()):
    print(
        f"  cluster {c}: "
        + "  ".join(f"{n} {p:.3f}" for n, p in zip(DEFAULT_POOL, row, strict=True))
    )

print("\nShare of requests answered by each arm:")
for n, s in zip(DEFAULT_POOL, model.selection_frequencies(data.X_test, n_draws=5), strict=True):
    print(f"  {n:<4} {s:6.1%}")
