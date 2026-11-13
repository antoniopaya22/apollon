# Guía rápida

## 1. Tráfico sintético, sin descargas

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

La última línea muestra la *mezcla de brazos*: la proporción de peticiones que responde cada
clasificador. Los mejores clasificadores responden con más frecuencia, pero todos conservan
alguna probabilidad de responder. `ApollonMAB(prior_strength=...)` mantiene una mezcla más
variada (consulta
[Cómo funciona Apollon](../guide/how-it-works.md#exploración-y-mezcla-de-brazos)).

## 2. En Python

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

## 3. Un conjunto de datos real

Descarga los ficheros *MachineLearningCSV* de
[CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) en
`data/CIC-IDS2017/` y, a continuación:

```bash
apollon prepare  --dataset cic-ids2017 --raw data/CIC-IDS2017 --out data/cic17.npz
apollon train    --data data/cic17.npz --out models/cic17.joblib
apollon evaluate --data data/cic17.npz --model models/cic17.joblib
apollon attack   --data data/cic17.npz --model models/cic17.joblib --attack hsja --n 100
```

`prepare --sample 200000` conserva un subconjunto estratificado para que la primera ejecución
sea más rápida. Consulta [Conjuntos de datos](../guide/datasets.md) para los demás conjuntos
de datos y [Línea de comandos](../guide/cli.md) para todas las opciones.

## 4. Realimentación

Cuando los analistas confirman o corrigen las alertas, devuelve al modelo la etiqueta real:

```python
pred, arms = model.predict(X_batch, return_arms=True)
clusters = model.assign_clusters(X_batch)
# ... later, when labels are known
model.update(X_batch, y_true, clusters=clusters, arms=arms, y_pred=pred)
```

Los brazos que acertaron pasan a tener más probabilidad de responder a las peticiones de ese
clúster.
