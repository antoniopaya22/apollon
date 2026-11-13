# Command line

Every command has `--help`. `python -m apollon` is equivalent to `apollon`.

```text
apollon prepare   preprocess a dataset into a .npz split
apollon train     train Apollon on a prepared dataset
apollon evaluate  detection metrics on the test split
apollon attack    adversarial evaluation with ZOO or HopSkipJump
apollon demo      train and evaluate on synthetic data (no download)
```

## `apollon prepare`

| Option | Default | Meaning |
|---|---|---|
| `--dataset` | required | `cic-ids2017`, `cse-cic-ids2018` or `cic-ddos2019` |
| `--raw` | required | CSV files or directories (all their `*.csv`) |
| `--out` | required | output `.npz` file |
| `--test-size` | `0.3` | test fraction |
| `--sample N` | all | keep a stratified subset of `N` rows |
| `--drop-duplicates` | off | drop duplicated rows before splitting |
| `--seed` | `0` | shuffling and split seed |

## `apollon train`

| Option | Default | Meaning |
|---|---|---|
| `--data` | required | prepared `.npz` |
| `--out` | required | output model (`.joblib`) |
| `--arms` | `RF DT NB LR MLP` | classifier pool; also `KNN`, `SVC` |
| `--clusters` | `2` | K-Means clusters |
| `--val-size` | `0.2` | validation fraction for the priors |
| `--prior-strength` | none | pseudo-observations per (cluster, arm) |
| `--seed` | `0` | seed of the split, K-Means, arms and draws |

## `apollon evaluate`

| Option | Default | Meaning |
|---|---|---|
| `--data`, `--model` | required | prepared dataset and trained model |
| `--arm-names` | pool order | names to print for the arms |
| `--json` | off | print one JSON line instead of a table |

## `apollon attack`

| Option | Default | Meaning |
|---|---|---|
| `--data`, `--model` | required | prepared dataset and trained model |
| `--attack` | `hsja` | `hsja` (HopSkipJump) or `zoo` |
| `--setting` | `direct` | `direct` (query Apollon) or `transfer` (surrogate) |
| `--surrogate` | `RF` | surrogate arm for the transfer setting |
| `--n` | `100` | attack flows to perturb |
| `--max-iter` | `100` | attack iterations |
| `--no-constraints` | off | do not restore functional features |
| `--seed`, `--json` | `0`, off | sampling seed and JSON output |

## `apollon demo`

Trains the default pool on 5,000 synthetic flows (`--samples`) and prints the metrics and
the arm mix. Useful to check an installation.
