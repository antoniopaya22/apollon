# Adversarial evaluation

Apollon is evaluated against **black-box evasion attacks**: the attacker can submit
inputs and observe the answers, but knows neither the models nor their parameters. This
is the realistic setting for an IDS deployed in a network.

!!! note "Requirements"
    Install the `attacks` extra: `pip install "apollon-ids[attacks]"`. Attacks are run
    with the [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox).

## Attacks

| Attack (`--attack`) | Kind | Uses | Reference |
|---|---|---|---|
| `zoo` | score-based | class probabilities | Chen et al., *ZOO: Zeroth Order Optimization Based Black-box Attacks*, AISec 2017 |
| `hsja` | decision-based | predicted label only | Chen, Jordan and Wainwright, *HopSkipJumpAttack*, IEEE S&P 2020 |

Both are run **targeted towards the benign class**: starting from real attack flows, they
search for the smallest perturbation that makes the IDS answer *benign*. Notebooks for
generative (GAN) attacks are also included in `research/attacks/`.

## Settings

=== "Direct"

    The attack queries Apollon itself. Every query goes through cluster assignment and
    Thompson-sampled arm selection, exactly as a request reaching the deployed IDS.

    ```python
    from apollon.attacks import apollon_classifier, run_attack
    clf = apollon_classifier(model, n_features=data.X_test.shape[1])
    x_adv = run_attack("hsja", clf, x_attack, max_iter=100)
    ```

=== "Transfer"

    The attacker trains a surrogate model, crafts adversarial examples against it and
    then submits them to Apollon.

    ```python
    from apollon import make_arm
    from apollon.attacks import model_classifier, run_attack
    surrogate = make_arm("RF", random_state=0).fit(data.X_train, data.y_train)
    x_adv = run_attack("zoo", model_classifier(surrogate), x_attack, max_iter=100)
    ```

## Functional features

An adversarial flow must still be a working attack. Some features cannot change without
breaking it: the destination port, packet and segment sizes, header lengths, initial
window sizes and packet counts of the attack itself. Apollon lists 27 such
*functional features* of CIC-IDS2017 in `apollon.attacks.FUNCTIONAL_FEATURES` and puts
their original values back after the attack, so only the remaining features are
perturbed:

```python
from apollon.attacks import functional_indexes, restore_functional

functional, modifiable, missing = functional_indexes(data.feature_names)
x_adv = restore_functional(x_adv, x_attack, functional)
```

## Measuring robustness

```python
import numpy as np
from apollon.metrics import evaluate

before = evaluate(np.ones(len(x_attack)), model.predict(x_attack))["detection_rate"]
after = evaluate(np.ones(len(x_adv)), model.predict(x_adv))["detection_rate"]
print(f"detection {before:.1%} -> {after:.1%}, evasion rate {1 - after:.1%}")
```

The command line runs the whole procedure:

```bash
apollon attack --data data/cic17.npz --model models/cic17.joblib \
               --attack hsja --setting direct --n 100 --max-iter 100
apollon attack --data data/cic17.npz --model models/cic17.joblib \
               --attack zoo --setting transfer --surrogate RF
```

!!! tip "Budget"
    Black-box attacks issue thousands of queries per sample. Start with `--n 20
    --max-iter 20` to check the setup, then increase.
