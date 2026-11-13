# Evaluación adversaria

Apollon se evalúa frente a **ataques de evasión de caja negra**: el atacante puede enviar
entradas y observar las respuestas, pero no conoce ni los modelos ni sus parámetros. Es el
escenario realista para un IDS desplegado en una red.

!!! note "Requisitos"
    Instala el extra `attacks`: `pip install "apollon-ids[attacks]"`. Los ataques se ejecutan
    con la [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox).

## Ataques

| Ataque (`--attack`) | Tipo | Utiliza | Referencia |
|---|---|---|---|
| `zoo` | basado en puntuaciones | probabilidades de clase | Chen et al., *ZOO: Zeroth Order Optimization Based Black-box Attacks*, AISec 2017 |
| `hsja` | basado en decisiones | solo la etiqueta predicha | Chen, Jordan y Wainwright, *HopSkipJumpAttack*, IEEE S&P 2020 |

Ambos se ejecutan de forma **dirigida hacia la clase benigna**: partiendo de flujos de ataque
reales, buscan la menor perturbación que hace que el IDS responda *benigno*. En
`research/attacks/` también se incluyen notebooks de ataques generativos (GAN).

## Escenarios

<figure class="ap-fig">
<svg viewBox="0 0 720 270" role="img" aria-label="Escenario directo: el atacante consulta a Apollon. Escenario de transferencia: el atacante genera ejemplos sobre un modelo sustituto y los envía a Apollon.">
<defs><marker id="ap-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#8e95c9"/></marker><marker id="ap-arr-r" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#f43f5e"/></marker></defs>
<text class="t" x="20" y="24">Directo</text>
<rect class="box-atk" x="20" y="40" width="130" height="70" rx="12"/><text class="t" x="85" y="70" text-anchor="middle">Atacante</text><text class="m" x="85" y="90" text-anchor="middle">ZOO / HopSkipJump</text>
<path class="arrow-q flow" d="M152,62 H418"/><text class="m" x="285" y="54" text-anchor="middle">consultas x + δ (miles)</text>
<path class="arrow" d="M418,90 H152"/><text class="m" x="285" y="106" text-anchor="middle">respuestas de un brazo distinto cada vez</text>
<rect class="box-hi" x="420" y="34" width="280" height="82" rx="12"/><text class="t" x="560" y="58" text-anchor="middle">Apollon</text>
<g><rect class="box" x="436" y="70" width="44" height="30" rx="7"/><text class="m" x="458" y="89" text-anchor="middle">RF</text><rect class="box" x="488" y="70" width="44" height="30" rx="7"/><text class="m" x="510" y="89" text-anchor="middle">DT</text><rect class="box" x="540" y="70" width="44" height="30" rx="7"/><text class="m" x="562" y="89" text-anchor="middle">NB</text><rect class="box" x="592" y="70" width="44" height="30" rx="7"/><text class="m" x="614" y="89" text-anchor="middle">LR</text><rect class="box" x="644" y="70" width="44" height="30" rx="7"/><text class="m" x="666" y="89" text-anchor="middle">MLP</text></g>
<text class="t" x="20" y="164">Transferencia</text>
<rect class="box-atk" x="20" y="180" width="130" height="70" rx="12"/><text class="t" x="85" y="210" text-anchor="middle">Atacante</text><text class="m" x="85" y="230" text-anchor="middle">ZOO / HopSkipJump</text>
<path class="arrow-q flow" d="M152,215 H228"/>
<rect class="box" x="230" y="185" width="130" height="60" rx="12"/><text class="t" x="295" y="210" text-anchor="middle">Sustituto</text><text class="m" x="295" y="228" text-anchor="middle">un solo clasificador</text>
<path class="arrow" d="M362,215 H418"/><text class="m" x="390" y="205" text-anchor="middle">x_adv</text>
<rect class="box-hi" x="420" y="180" width="280" height="70" rx="12"/><text class="t" x="560" y="210" text-anchor="middle">Apollon</text><text class="m" x="560" y="230" text-anchor="middle">evalúa los ejemplos adversarios</text>
</svg>
<figcaption>Los dos escenarios de la evaluación adversaria. En ambos, las características funcionales se restauran después del ataque.</figcaption>
</figure>


=== "Directo"

    El ataque consulta directamente a Apollon. Cada consulta pasa por la asignación de clúster
    y por la selección de brazo mediante muestreo de Thompson, exactamente igual que una
    petición que llega al IDS desplegado.

    ```python
    from apollon.attacks import apollon_classifier, run_attack
    clf = apollon_classifier(model, n_features=data.X_test.shape[1])
    x_adv = run_attack("hsja", clf, x_attack, max_iter=100)
    ```

=== "Transferencia"

    El atacante entrena un modelo sustituto, genera ejemplos adversarios contra él y después
    los envía a Apollon.

    ```python
    from apollon import make_arm
    from apollon.attacks import model_classifier, run_attack
    surrogate = make_arm("RF", random_state=0).fit(data.X_train, data.y_train)
    x_adv = run_attack("zoo", model_classifier(surrogate), x_attack, max_iter=100)
    ```

## Características funcionales

Un flujo adversario tiene que seguir siendo un ataque operativo. Algunas características no
pueden cambiar sin romperlo: el puerto de destino, los tamaños de paquete y de segmento, las
longitudes de cabecera, los tamaños iniciales de ventana y los recuentos de paquetes del
propio ataque. Apollon enumera 27 de estas *características funcionales* de CIC-IDS2017 en
`apollon.attacks.FUNCTIONAL_FEATURES` y restaura sus valores originales después del ataque,
de modo que solo se perturban las características restantes:

```python
from apollon.attacks import functional_indexes, restore_functional

functional, modifiable, missing = functional_indexes(data.feature_names)
x_adv = restore_functional(x_adv, x_attack, functional)
```

## Medir la robustez

```python
import numpy as np
from apollon.metrics import evaluate

before = evaluate(np.ones(len(x_attack)), model.predict(x_attack))["detection_rate"]
after = evaluate(np.ones(len(x_adv)), model.predict(x_adv))["detection_rate"]
print(f"detection {before:.1%} -> {after:.1%}, evasion rate {1 - after:.1%}")
```

La línea de comandos ejecuta el procedimiento completo:

```bash
apollon attack --data data/cic17.npz --model models/cic17.joblib \
               --attack hsja --setting direct --n 100 --max-iter 100
apollon attack --data data/cic17.npz --model models/cic17.joblib \
               --attack zoo --setting transfer --surrogate RF
```

!!! tip "Presupuesto"
    Los ataques de caja negra lanzan miles de consultas por muestra. Empieza con `--n 20
    --max-iter 20` para comprobar la configuración y, después, auméntalo.
