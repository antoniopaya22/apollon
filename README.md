# Apollon

[![Paper](https://img.shields.io/badge/DOI-10.1016%2Fj.cose.2023.103546-0E7490)](https://doi.org/10.1016/j.cose.2023.103546)
[![Journal](https://img.shields.io/badge/Computers%20%26%20Security-2024-4D7C0F)](https://doi.org/10.1016/j.cose.2023.103546)
[![License](https://img.shields.io/github/license/antoniopaya22/apollon)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10-3776AB?logo=python&logoColor=white)

Official implementation of **[Apollon: A robust defense system against Adversarial Machine Learning attacks in Intrusion Detection Systems](https://doi.org/10.1016/j.cose.2023.103546)**, published in *Computers & Security*, vol. 136, 103546 (2024).

## Abstract

The rise of Adversarial Machine Learning (AML) attacks is presenting a significant
challenge to Intrusion Detection Systems (IDS) and their ability to detect threats. To
address this issue, we introduce Apollon, a novel defence system that can protect IDS
against AML attacks. Apollon utilizes a diverse set of classifiers to identify intrusions
and employs Multi-Armed Bandits (MAB) with Thompson sampling to dynamically
select the optimal classifier or ensemble of classifiers for each input. This approach
enables Apollon to prevent attackers from learning the IDS behaviour and generating
adversarial examples that can evade the IDS detection. We evaluate Apollon on
several of the most popular and recent datasets, and show that it can successfully
detect attacks without compromising its performance on traditional network traffic. Our
results suggest that Apollon is a robust defence system against AML attacks in IDS.

## How it works

1. A pool of diverse classifiers (and ensembles) is trained to detect intrusions.
2. For every input, a **Multi-Armed Bandit with Thompson sampling** selects which classifier or ensemble answers.
3. Because the decision boundary keeps changing, an attacker cannot learn a stable model of the IDS to craft adversarial examples.

## Repository structure

```text
attacks/                       # adversarial example generation
datasets/                      # dataset preprocessing (see datasets/README.md)
intrusion_detection_systems/   # IDS models, training and metrics
MAB/                           # Multi-Armed Bandit selection strategies
results/                       # experiment outputs
shared/                        # utilities, models and data folder
main.py                        # preprocessing + IDS training pipeline
mab.ipynb                      # Apollon (MAB) experiments
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

1. Download the datasets used in the paper ([CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html), [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html), [CIC-DDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html), [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)). For CIC-IDS2017, place the CSV files under `shared/data/CIC_2017/`.
2. The first time, set `load_dataset = False` in `main.py` so the raw CSVs are preprocessed and saved, then run:

   ```bash
   python main.py
   ```

3. Open `mab.ipynb` to run the Apollon experiments on top of the trained models.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{paya2024apollon,
  title   = {Apollon: A robust defense system against Adversarial Machine Learning attacks in Intrusion Detection Systems},
  author  = {Pay{\'a}, Antonio and Arroni, Sergio and Garc{\'i}a-D{\'i}az, Vicente and G{\'o}mez, Alberto},
  journal = {Computers \& Security},
  volume  = {136},
  pages   = {103546},
  year    = {2024},
  doi     = {10.1016/j.cose.2023.103546}
}
```

## License

[Apache 2.0](LICENSE)
