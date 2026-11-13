---
template: home.html
title: Apollon
hide:
  - navigation
  - toc
---

# Apollon

**A robust defence system against adversarial machine learning attacks in intrusion detection systems.**

[![Paper](https://img.shields.io/badge/DOI-10.1016%2Fj.cose.2023.103546-0E7490)](https://doi.org/10.1016/j.cose.2023.103546)
[![Journal](https://img.shields.io/badge/Computers%20%26%20Security-vol.%20136%20(2024)-4D7C0F)](https://doi.org/10.1016/j.cose.2023.103546)
[![CI](https://github.com/antoniopaya22/apollon/actions/workflows/ci.yml/badge.svg)](https://github.com/antoniopaya22/apollon/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](about/license.md)

Machine-learning intrusion detection systems (IDS) are accurate on ordinary traffic, but
an attacker who can query them can learn their decision boundary and craft *adversarial
examples*: attacks perturbed just enough to be classified as benign. Apollon protects an
IDS by never giving the attacker a fixed model to learn. Every request is answered by a
classifier drawn from a diverse pool by a **Multi-Armed Bandit with Thompson sampling**,
so the decision boundary the attacker probes keeps changing, while the bandit still
favours the classifiers that perform best on each kind of traffic.

<div class="grid cards" markdown>

-   :material-rocket-launch: **Get started in a minute**

    ---

    Install the package and train Apollon on synthetic traffic, no download needed.

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

-   :material-brain: **How it works**

    ---

    Clustering, a pool of classifiers per cluster and Thompson sampling with Beta
    posteriors that keep learning from feedback.

    [:octicons-arrow-right-24: How Apollon works](guide/how-it-works.md)

-   :material-sword-cross: **Adversarial evaluation**

    ---

    Black-box ZOO and HopSkipJump attacks, in transfer and direct settings, with
    functional-feature constraints.

    [:octicons-arrow-right-24: Adversarial evaluation](guide/attacks.md)

-   :material-flask: **Experiments**

    ---

    CIC-IDS2017, CSE-CIC-IDS2018 and CIC-DDoS2019, the classifier pool of the paper and
    the notebooks that produced its results.

    [:octicons-arrow-right-24: Experiments](experiments/index.md)

</div>

## At a glance

```python
from apollon import ApollonMAB
from apollon.data import prepare

data = prepare("cic-ids2017", ["data/CIC-IDS2017/"])          # CSVs -> scaled split
model = ApollonMAB(random_state=0).fit(data.X_train, data.y_train)
labels, arms = model.predict(data.X_test, return_arms=True)   # one draw per request
```

```console
$ pip install "apollon-ids[attacks] @ git+https://github.com/antoniopaya22/apollon"
$ apollon demo
```

## Highlights

- **Moving-target defence for IDS.** The classifier that answers changes from request to
  request, which prevents attackers from learning a stable model to evade.
- **Detection is not compromised.** Thompson sampling concentrates on the classifiers
  that perform best on each cluster of traffic, so Apollon detects attacks without
  compromising its performance on ordinary network traffic.
- **Keeps learning.** Analyst feedback updates the Beta posteriors online
  ([`update`](reference/mab.md)).
- **Scikit-learn compatible.** `fit`, `predict`, `predict_proba`, `get_params`; any
  scikit-learn classifier can join the pool.
- **Reproducible.** Fixed seeds, a command line interface and the original notebooks of
  the paper.

## Citing

If Apollon helps your research, please cite the paper:

```bibtex
--8<-- "docs/about/apollon.bib"
```

See [Citing Apollon](about/citation.md) for other formats.
