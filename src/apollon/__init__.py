"""Apollon: a robust defence system against adversarial machine learning attacks in IDS.

Apollon answers every request with a classifier chosen from a diverse pool by a
per-cluster Multi-Armed Bandit with Thompson sampling, so attackers cannot learn a stable
model of the IDS to craft adversarial examples against.

Paper: A. Payá, S. Arroni, V. García-Díaz, A. Gómez. *Apollon: A robust defense system
against Adversarial Machine Learning attacks in Intrusion Detection Systems.* Computers &
Security 136, 103546 (2024). https://doi.org/10.1016/j.cose.2023.103546
"""

from apollon.mab import ApollonMAB, load_model, save_model
from apollon.pool import ARMS, DEFAULT_POOL, default_arms, make_arm, make_pool

__version__ = "2.0.0"

__all__ = [
    "ARMS",
    "DEFAULT_POOL",
    "ApollonMAB",
    "__version__",
    "default_arms",
    "load_model",
    "make_arm",
    "make_pool",
    "save_model",
]
