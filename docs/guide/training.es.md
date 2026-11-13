# Entrenamiento y evaluación

## Entrenamiento

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

o bien

```bash
apollon train --data data/cic17.npz --out models/cic17.joblib --arms RF DT NB LR MLP --clusters 2
```

`fit` separa el conjunto de validación, ajusta K-Means, entrena cada brazo en cada clúster e
inicializa las posteriores Beta a partir de la exactitud de validación de cada brazo en cada
clúster. Un clúster que contiene una sola clase no puede entrenar un clasificador por sí
mismo; en ese caso, el clúster usa el brazo entrenado con todos los datos (`model.shared_`
marca esos casos).

## Predicción

```python
labels = model.predict(X)                         # one Thompson draw per row
labels, arms = model.predict(X, return_arms=True) # and which arm answered
proba = model.predict_proba(X)                    # probabilities of the arm that answered
```

Las predicciones son estocásticas por diseño: la misma fila puede ser respondida por brazos
distintos en llamadas distintas. Fijar la semilla (`random_state`) hace reproducible una
secuencia completa de llamadas.

!!! warning "Calcula las métricas a partir de una única predicción"
    Como cada llamada extrae brazos nuevos, calcula la exactitud, la tasa de detección y el F1
    a partir de la misma llamada a `predict`, como hace `apollon.metrics.evaluate`, en lugar
    de llamar a `predict` una vez por métrica.

## Evaluación

```python
from apollon.metrics import evaluate, format_metrics

pred = model.predict(data.X_test)
print(format_metrics(evaluate(data.y_test, pred)))
```

| Métrica | Definición |
|---|---|
| `accuracy` | proporción de respuestas correctas (exactitud) |
| `detection_rate` | sensibilidad sobre los ataques (tasa de verdaderos positivos) |
| `f1` | puntuación F1 de la clase de ataque |
| `false_positive_rate` | proporción de flujos benignos marcados como ataques |
| `roc_auc` | área bajo la curva ROC (a partir de las etiquetas, o de las puntuaciones si se proporcionan) |

```bash
apollon evaluate --data data/cic17.npz --model models/cic17.joblib          # table
apollon evaluate --data data/cic17.npz --model models/cic17.joblib --json   # one JSON line
```

## Realimentación en línea

```python
pred, arms = model.predict(X_batch, return_arms=True)
clusters = model.assign_clusters(X_batch)
model.update(X_batch, y_true, clusters=clusters, arms=arms, y_pred=pred)
```

Pasar los `clusters`, los `arms` y las predicciones de la respuesta original acredita
exactamente a los brazos que respondieron a esas peticiones.

## Persistencia

`save_model` y `load_model` usan joblib. Carga solo ficheros de modelo de confianza: como
cualquier formato basado en pickle, cargarlos puede ejecutar código.
