# Referencia de la API

| Módulo | Contenido |
|---|---|
| [`apollon.mab`](mab.md) | `ApollonMAB`, el estimador; `save_model`, `load_model` |
| [`apollon.pool`](pool.md) | El conjunto de clasificadores: `ARMS`, `DEFAULT_POOL`, `make_pool` |
| [`apollon.data`](data.md) | Especificaciones de conjuntos de datos, limpieza, partición y escalado |
| [`apollon.attacks`](attacks.md) | Envoltorios de ART, ZOO y HopSkipJump, características funcionales |
| [`apollon.metrics`](metrics.md) | Métricas de detección |

Los nombres más usados se reexportan en el nivel superior del paquete:

```python
from apollon import ApollonMAB, make_pool, default_arms, save_model, load_model
```
