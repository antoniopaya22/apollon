# Datasets

Apollon works on any table of flow features with a binary label. The package includes
loaders for the three datasets of the Canadian Institute for Cybersecurity used in the
paper. Datasets are not redistributed: download them from their official pages.

| Name (`--dataset`) | Dataset | Files to download | Label column |
|---|---|---|---|
| `cic-ids2017` | [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) | `MachineLearningCSV.zip` (8 CSV files) | ` Label` (`BENIGN` / attack name) |
| `cse-cic-ids2018` | [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html) | `Processed Traffic Data for ML Algorithms` (10 CSV files, from the AWS bucket) | `Label` (`Benign` / attack name) |
| `cic-ddos2019` | [CIC-DDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html) | `CSV-01-12.zip` and/or `CSV-03-11.zip` | ` Label` (`BENIGN` / attack name) |

!!! info "Column names"
    The CIC files keep a leading space in many column names (`" Label"`,
    `" Destination Port"`). Apollon keeps the original names, so the functional-feature
    list of the [adversarial evaluation](attacks.md) matches them.

## Preprocessing

`apollon prepare` (or [`apollon.data.prepare`](../reference/data.md)) runs the
pipeline of the paper:

1. read and concatenate the CSV files (a directory expands to its `*.csv` files) and
   shuffle them with a fixed seed;
2. drop identifier columns (flow id, addresses, ports and timestamps where the dataset
   has them), header lines repeated inside the files and constant features;
3. replace infinite and missing values by 0 and map labels to 0 (benign) or 1 (any
   attack);
4. split 70/30 into train and test, stratified by label;
5. scale with `RobustScaler(with_centering=False)` and `Normalizer`, both fitted on the
   training split only.

```bash
apollon prepare --dataset cse-cic-ids2018 --raw data/CSE-CIC-IDS2018 \
                --out data/cic18.npz --sample 500000
```

The output is a single compressed `.npz` file with the train/test split, the feature
names and the dataset name, loaded with `apollon.data.Dataset.load`.

!!! tip "Memory"
    CSE-CIC-IDS2018 has about 16 million flows. Use `--sample` to keep a stratified
    subset, or prepare it on a machine with at least 32 GB of RAM.

## Your own data

Any `pandas` table works through the lower-level functions:

```python
import pandas as pd
from apollon.data import DatasetSpec, clean, split_and_scale

spec = DatasetSpec(name="My network", label="label", benign="normal",
                   drop=("src_ip", "dst_ip", "timestamp"))
X, y = clean(pd.read_csv("flows.csv"), spec)
data = split_and_scale(X, y, seed=0)
data.save("data/my-network.npz")       # usable by `apollon train`
```
