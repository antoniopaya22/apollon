# Quickstart

## 1. Synthetic traffic, no download

```console
$ apollon demo
synthetic: 20 features, 3500 train (20.2% attacks), 1500 test (20.2% attacks)

Apollon on synthetic traffic
----------------------------
accuracy             0.9587
detection_rate       0.8086
f1                   0.8877
false_positive_rate  0.0033
roc_auc              0.9026

Arm selection: RF 79%, DT 1%, NB 0%, LR 19%, MLP 0%
```

The last line shows the *arm mix*: the share of requests answered by each classifier.
Better classifiers answer more often, but every classifier keeps a chance of answering.
`ApollonMAB(prior_strength=...)` keeps the mix wider (see
[How Apollon works](../guide/how-it-works.md#exploration-and-the-arm-mix)).

## 2. In Python

```python
from sklearn.datasets import make_classification
from apollon import ApollonMAB
from apollon.data import split_and_scale
from apollon.metrics import evaluate, format_metrics

X, y = make_classification(n_samples=5000, n_features=20, n_informative=12,
                           weights=[0.8, 0.2], random_state=0)
data = split_and_scale(X, y, seed=0, normalize=False)

model = ApollonMAB(n_clusters=2, random_state=0).fit(data.X_train, data.y_train)
pred, arms = model.predict(data.X_test, return_arms=True)
print(format_metrics(evaluate(data.y_test, pred)))
print(model.selection_frequencies(data.X_test))     # share of each arm
print(model.posterior_means())                      # expected accuracy per cluster and arm
```

## 3. A real dataset

Download the *MachineLearningCSV* files of
[CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) into
`data/CIC-IDS2017/`, then:

```bash
apollon prepare  --dataset cic-ids2017 --raw data/CIC-IDS2017 --out data/cic17.npz
apollon train    --data data/cic17.npz --out models/cic17.joblib
apollon evaluate --data data/cic17.npz --model models/cic17.joblib
apollon attack   --data data/cic17.npz --model models/cic17.joblib --attack hsja --n 100
```

`prepare --sample 200000` keeps a stratified subset for a quicker first run. See
[Datasets](../guide/datasets.md) for the other datasets and
[Command line](../guide/cli.md) for every option.

## 4. Feedback

When analysts confirm or correct alerts, feed the ground truth back:

```python
pred, arms = model.predict(X_batch, return_arms=True)
clusters = model.assign_clusters(X_batch)
# ... later, when labels are known
model.update(X_batch, y_true, clusters=clusters, arms=arms, y_pred=pred)
```

The arms that answered correctly become more likely to answer requests of that cluster.
