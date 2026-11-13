# API reference

| Module | Content |
|---|---|
| [`apollon.mab`](mab.md) | `ApollonMAB`, the estimator; `save_model`, `load_model` |
| [`apollon.pool`](pool.md) | The classifier pool: `ARMS`, `DEFAULT_POOL`, `make_pool` |
| [`apollon.data`](data.md) | Dataset specifications, cleaning, splitting and scaling |
| [`apollon.attacks`](attacks.md) | ART wrappers, ZOO and HopSkipJump, functional features |
| [`apollon.metrics`](metrics.md) | Detection metrics |

The most used names are re-exported at the top level:

```python
from apollon import ApollonMAB, make_pool, default_arms, save_model, load_model
```
