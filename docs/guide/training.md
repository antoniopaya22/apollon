# Training and evaluation

## Training

```python
from apollon import ApollonMAB, make_pool, save_model
from apollon.data import Dataset

data = Dataset.load("data/cic17.npz")
model = ApollonMAB(
    arms=make_pool(["RF", "DT", "NB", "LR", "MLP"], random_state=0),  # the paper's pool
    n_clusters=2,          # K-Means clusters
    val_size=0.2,          # held out to initialise the posteriors
    prior_strength=None,   # see "Exploration and the arm mix"
    random_state=0,
).fit(data.X_train, data.y_train)
save_model(model, "models/cic17.joblib")
```

or

```bash
apollon train --data data/cic17.npz --out models/cic17.joblib --arms RF DT NB LR MLP --clusters 2
```

`fit` splits off the validation set, fits K-Means, trains every arm on every cluster and
initialises the Beta posteriors from each arm's validation accuracy on each cluster. A
cluster that contains a single class cannot train a classifier on its own; that cluster
then uses the arm trained on all the data (`model.shared_` marks those cases).

## Prediction

```python
labels = model.predict(X)                         # one Thompson draw per row
labels, arms = model.predict(X, return_arms=True) # and which arm answered
proba = model.predict_proba(X)                    # probabilities of the arm that answered
```

Predictions are stochastic by design: the same row can be answered by different arms in
different calls. Seeding (`random_state`) makes a whole sequence of calls reproducible.

!!! warning "Compute metrics from a single prediction"
    Because every call draws new arms, compute accuracy, detection rate and F1 from the
    same call to `predict`, as `apollon.metrics.evaluate` does, rather than calling
    `predict` once per metric.

## Evaluation

```python
from apollon.metrics import evaluate, format_metrics

pred = model.predict(data.X_test)
print(format_metrics(evaluate(data.y_test, pred)))
```

| Metric | Definition |
|---|---|
| `accuracy` | share of correct answers |
| `detection_rate` | recall on attacks (true positive rate) |
| `f1` | F1 score of the attack class |
| `false_positive_rate` | share of benign flows flagged as attacks |
| `roc_auc` | area under the ROC curve (from labels, or from scores if given) |

```bash
apollon evaluate --data data/cic17.npz --model models/cic17.joblib          # table
apollon evaluate --data data/cic17.npz --model models/cic17.joblib --json   # one JSON line
```

## Online feedback

```python
pred, arms = model.predict(X_batch, return_arms=True)
clusters = model.assign_clusters(X_batch)
model.update(X_batch, y_true, clusters=clusters, arms=arms, y_pred=pred)
```

Passing the `clusters`, `arms` and predictions of the original answer credits exactly
the arms that answered those requests.

## Persistence

`save_model` and `load_model` use joblib. Only load model files you trust: like any
pickle-based format, loading can execute code.
