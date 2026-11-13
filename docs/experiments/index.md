# Experiments

The paper evaluates Apollon on three public datasets, against black-box adversarial
attacks, and compares it with the classifiers of its pool used alone.

## Setup

| | |
|---|---|
| **Datasets** | CIC-IDS2017, CSE-CIC-IDS2018, CIC-DDoS2019 ([Datasets](../guide/datasets.md)) |
| **Split** | 70 % train / 30 % test, stratified; 20 % of train held out for the posteriors |
| **Preprocessing** | constant and identifier features removed, `RobustScaler` + `Normalizer` |
| **Pool** | Random Forest, Decision Tree, Gaussian NB, Logistic Regression, MLP |
| **Clusters** | 2 (K-Means) |
| **Baselines** | each classifier of the pool alone |
| **Attacks** | ZOO and HopSkipJump, targeted to benign, 100 iterations, functional features fixed |
| **Metrics** | accuracy, detection rate, F1, ROC AUC ([definitions](../guide/training.md#evaluation)) |
| **Seed** | 0 for splits, models and Thompson draws |

## Questions

1. **Detection on ordinary traffic.** Does selecting classifiers with Thompson sampling
   keep the detection performance of the best classifiers of the pool?
2. **Robustness to adversarial examples.** How many adversarial examples crafted with
   ZOO and HopSkipJump does Apollon still detect, compared with the single classifiers?

The answers, with all the numbers, are in the
[paper](https://doi.org/10.1016/j.cose.2023.103546). The next pages explain how to
[reproduce](reproduce.md) them and what the [recorded outputs](results.md) of the
repository contain.
