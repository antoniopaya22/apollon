# Research code of the paper

The scripts and notebooks that produced the results of *Apollon: A robust defense system
against Adversarial Machine Learning attacks in Intrusion Detection Systems* (Computers &
Security, 2024). New functionality lives in the package (`../src/apollon`); the notebooks
import it.

| Path | Content |
|---|---|
| `main.py` | Preprocessing of CIC-IDS2017 and training of the single classifiers |
| `mab.ipynb`, `MAB/` | Apollon (per-cluster Thompson sampling) on a prepared dataset |
| `attacks/zoo.ipynb`, `attacks/hotskipjump.ipynb` | Black-box attacks against the classifiers and against Apollon |
| `attacks/gan.ipynb`, `attacks/GAN*.py` | Generative adversarial examples |
| `datasets/`, `shared/` | Preprocessing of CIC-IDS2017, CSE-CIC-IDS2018 and CIC-DDoS2019 |
| `intrusion_detection_systems/` | Training and metrics of the single classifiers |
| `results/` | Recorded outputs of the runs |
| `requirements.txt` | Versions used for the paper |

## Running

```bash
# from the repository root
pip install -e ".[attacks]"
pip install -r research/requirements.txt   # optional: exact versions of the paper
cd research
python main.py                             # set load_dataset = False the first time
jupyter lab mab.ipynb
```

Place the CIC-IDS2017 CSV files under `research/shared/data/CIC_2017/`. Run the notebooks
from this folder so that `MAB/`, `attacks/`, `datasets/` and `shared/` are importable.

See the [documentation](https://antoniopaya22.github.io/apollon/experiments/reproduce/)
for the equivalent command line pipeline.
