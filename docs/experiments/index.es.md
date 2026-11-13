# Experimentos

El artículo evalúa Apollon con tres conjuntos de datos públicos, frente a ataques adversarios
de caja negra, y lo compara con los clasificadores de su conjunto usados por separado.

## Configuración

| | |
|---|---|
| **Conjuntos de datos** | CIC-IDS2017, CSE-CIC-IDS2018, CIC-DDoS2019 ([Conjuntos de datos](../guide/datasets.md)) |
| **Partición** | 70 % entrenamiento / 30 % prueba, estratificada; el 20 % del entrenamiento se reserva para las posteriores |
| **Preprocesamiento** | se eliminan las características constantes e identificadoras, `RobustScaler` + `Normalizer` |
| **Conjunto de clasificadores** | Random Forest, Decision Tree, Gaussian NB, Logistic Regression, MLP |
| **Clústeres** | 2 (K-Means) |
| **Referencias de comparación** | cada clasificador del conjunto por separado |
| **Ataques** | ZOO y HopSkipJump, dirigidos a benigno, 100 iteraciones, características funcionales fijas |
| **Métricas** | exactitud, tasa de detección, F1, ROC AUC ([definiciones](../guide/training.md#evaluación)) |
| **Semilla** | 0 para las particiones, los modelos y las extracciones de Thompson |

## Preguntas

1. **Detección en tráfico ordinario.** ¿Seleccionar clasificadores con muestreo de Thompson
   mantiene el rendimiento de detección de los mejores clasificadores del conjunto?
2. **Robustez frente a ejemplos adversarios.** ¿Cuántos ejemplos adversarios generados con
   ZOO y HopSkipJump sigue detectando Apollon, en comparación con los clasificadores
   individuales?

Las respuestas, con todas las cifras, están en el
[artículo](https://doi.org/10.1016/j.cose.2023.103546). Las páginas siguientes explican cómo
[reproducirlas](reproduce.md) y qué contienen las [salidas registradas](results.md) del
repositorio.
