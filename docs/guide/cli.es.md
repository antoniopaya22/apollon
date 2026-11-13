# Línea de comandos

Todos los comandos admiten `--help`. `python -m apollon` equivale a `apollon`.

```text
apollon prepare   preprocess a dataset into a .npz split
apollon train     train Apollon on a prepared dataset
apollon evaluate  detection metrics on the test split
apollon attack    adversarial evaluation with ZOO or HopSkipJump
apollon demo      train and evaluate on synthetic data (no download)
```

## `apollon prepare`

| Opción | Por defecto | Significado |
|---|---|---|
| `--dataset` | obligatoria | `cic-ids2017`, `cse-cic-ids2018` o `cic-ddos2019` |
| `--raw` | obligatoria | ficheros CSV o directorios (todos sus `*.csv`) |
| `--out` | obligatoria | fichero `.npz` de salida |
| `--test-size` | `0.3` | fracción de prueba |
| `--sample N` | todas | conserva un subconjunto estratificado de `N` filas |
| `--drop-duplicates` | desactivada | elimina las filas duplicadas antes de dividir |
| `--seed` | `0` | semilla del barajado y de la partición |

## `apollon train`

| Opción | Por defecto | Significado |
|---|---|---|
| `--data` | obligatoria | `.npz` preparado |
| `--out` | obligatoria | modelo de salida (`.joblib`) |
| `--arms` | `RF DT NB LR MLP` | conjunto de clasificadores; también `KNN`, `SVC` |
| `--clusters` | `2` | clústeres de K-Means |
| `--val-size` | `0.2` | fracción de validación para las distribuciones a priori |
| `--prior-strength` | ninguno | pseudoobservaciones por (clúster, brazo) |
| `--seed` | `0` | semilla de la partición, K-Means, los brazos y las extracciones |

## `apollon evaluate`

| Opción | Por defecto | Significado |
|---|---|---|
| `--data`, `--model` | obligatorias | conjunto de datos preparado y modelo entrenado |
| `--arm-names` | orden del conjunto | nombres que mostrar para los brazos |
| `--json` | desactivada | imprime una línea JSON en lugar de una tabla |

## `apollon attack`

| Opción | Por defecto | Significado |
|---|---|---|
| `--data`, `--model` | obligatorias | conjunto de datos preparado y modelo entrenado |
| `--attack` | `hsja` | `hsja` (HopSkipJump) o `zoo` |
| `--setting` | `direct` | `direct` (consulta a Apollon) o `transfer` (modelo sustituto) |
| `--surrogate` | `RF` | brazo sustituto para el escenario de transferencia |
| `--n` | `100` | flujos de ataque que perturbar |
| `--max-iter` | `100` | iteraciones del ataque |
| `--no-constraints` | desactivada | no restaura las características funcionales |
| `--seed`, `--json` | `0`, desactivada | semilla del muestreo y salida JSON |

## `apollon demo`

Entrena el conjunto por defecto con 5000 flujos sintéticos (`--samples`) y muestra las
métricas y la mezcla de brazos. Resulta útil para comprobar una instalación.
