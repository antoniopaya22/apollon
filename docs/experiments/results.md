# Recorded outputs

`research/results/` keeps the raw outputs of the experiment runs of the paper, as they
were printed by the scripts and notebooks.

| File | Content |
|---|---|
| `apollon_results.txt` | Training of the single classifiers on CIC-IDS2017: cross-validated accuracy, detection rate, F1 and ROC AUC on train and test, confusion matrices, classification reports and training and prediction times |
| `2018.txt` | The same for CSE-CIC-IDS2018 |
| `2019.txt` | The same for CIC-DDoS2019 |
| `MAB.txt` | Classification report of an Apollon run |
| `Apollon.xlsx` | Tables of results collected for the paper |
| `arm_selections.log` | Arm selected, prediction and true label of every request of an Apollon run |

The definitive numbers, with the comparison against the single classifiers and the
adversarial evaluation, are those of the
[paper](https://doi.org/10.1016/j.cose.2023.103546). To regenerate results with the
current package, follow [Reproducing the paper](reproduce.md); `--json` outputs are easy
to collect into tables.
