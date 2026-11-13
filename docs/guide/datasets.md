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

<figure class="ap-fig">
<svg viewBox="0 0 720 120" role="img" aria-label="Preprocessing pipeline: CSV files, clean, labels, split, scale, npz.">
<defs><marker id="ap-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#8e95c9"/></marker><marker id="ap-arr-r" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#f43f5e"/></marker></defs>
<rect class="box-hi" x="10" y="30" width="100" height="64" rx="12"/><text class="t" x="60" y="58" text-anchor="middle">CSV files</text><text class="m" x="60" y="78" text-anchor="middle">raw CIC exports</text><path class="arrow flow" d="M112,62 H128"/><rect class="box" x="130" y="30" width="100" height="64" rx="12"/><text class="t" x="180" y="58" text-anchor="middle">Clean</text><text class="m" x="180" y="78" text-anchor="middle">ids, constants out</text><path class="arrow flow" d="M232,62 H248"/><rect class="box" x="250" y="30" width="100" height="64" rx="12"/><text class="t" x="300" y="58" text-anchor="middle">Labels</text><text class="m" x="300" y="78" text-anchor="middle">0 benign · 1 attack</text><path class="arrow flow" d="M352,62 H368"/><rect class="box" x="370" y="30" width="100" height="64" rx="12"/><text class="t" x="420" y="58" text-anchor="middle">Split</text><text class="m" x="420" y="78" text-anchor="middle">70/30 stratified</text><path class="arrow flow" d="M472,62 H488"/><rect class="box" x="490" y="30" width="100" height="64" rx="12"/><text class="t" x="540" y="58" text-anchor="middle">Scale</text><text class="m" x="540" y="78" text-anchor="middle">robust + unit norm</text><path class="arrow flow" d="M592,62 H608"/><rect class="box-hi" x="610" y="30" width="100" height="64" rx="12"/><text class="t" x="660" y="58" text-anchor="middle">.npz</text><text class="m" x="660" y="78" text-anchor="middle">portable split</text>
</svg>
<figcaption><code>apollon prepare</code>: from the raw CSV files of a dataset to a portable train/test split. Scalers are fitted on the training split only.</figcaption>
</figure>


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
