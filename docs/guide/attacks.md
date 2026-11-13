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

<figure class="ap-fig">
<svg viewBox="0 0 720 270" role="img" aria-label="Direct setting: the attacker queries Apollon. Transfer setting: the attacker crafts examples on a surrogate and submits them to Apollon.">
<defs><marker id="ap-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#8e95c9"/></marker><marker id="ap-arr-r" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#f43f5e"/></marker></defs>
<text class="t" x="20" y="24">Direct</text>
<rect class="box-atk" x="20" y="40" width="130" height="70" rx="12"/><text class="t" x="85" y="70" text-anchor="middle">Attacker</text><text class="m" x="85" y="90" text-anchor="middle">ZOO / HopSkipJump</text>
<path class="arrow-q flow" d="M152,62 H418"/><text class="m" x="285" y="54" text-anchor="middle">queries x + δ (thousands)</text>
<path class="arrow" d="M418,90 H152"/><text class="m" x="285" y="106" text-anchor="middle">answers from a different arm each time</text>
<rect class="box-hi" x="420" y="34" width="280" height="82" rx="12"/><text class="t" x="560" y="58" text-anchor="middle">Apollon</text>
<g><rect class="box" x="436" y="70" width="44" height="30" rx="7"/><text class="m" x="458" y="89" text-anchor="middle">RF</text><rect class="box" x="488" y="70" width="44" height="30" rx="7"/><text class="m" x="510" y="89" text-anchor="middle">DT</text><rect class="box" x="540" y="70" width="44" height="30" rx="7"/><text class="m" x="562" y="89" text-anchor="middle">NB</text><rect class="box" x="592" y="70" width="44" height="30" rx="7"/><text class="m" x="614" y="89" text-anchor="middle">LR</text><rect class="box" x="644" y="70" width="44" height="30" rx="7"/><text class="m" x="666" y="89" text-anchor="middle">MLP</text></g>
<text class="t" x="20" y="164">Transfer</text>
<rect class="box-atk" x="20" y="180" width="130" height="70" rx="12"/><text class="t" x="85" y="210" text-anchor="middle">Attacker</text><text class="m" x="85" y="230" text-anchor="middle">ZOO / HopSkipJump</text>
<path class="arrow-q flow" d="M152,215 H228"/>
<rect class="box" x="230" y="185" width="130" height="60" rx="12"/><text class="t" x="295" y="210" text-anchor="middle">Surrogate</text><text class="m" x="295" y="228" text-anchor="middle">single classifier</text>
<path class="arrow" d="M362,215 H418"/><text class="m" x="390" y="205" text-anchor="middle">x_adv</text>
<rect class="box-hi" x="420" y="180" width="280" height="70" rx="12"/><text class="t" x="560" y="210" text-anchor="middle">Apollon</text><text class="m" x="560" y="230" text-anchor="middle">evaluates the adversarial examples</text>
</svg>
<figcaption>The two settings of the adversarial evaluation. Functional features are restored after the attack in both.</figcaption>
</figure>


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
