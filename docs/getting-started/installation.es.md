# Instalación

Apollon necesita **Python 3.10 o posterior**. Solo depende de NumPy, pandas, scikit-learn
y joblib; la evaluación adversaria añade la
[Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox).

## Desde GitHub

=== "pip"

    ```bash
    python -m venv .venv && source .venv/bin/activate
    pip install "apollon-ids @ git+https://github.com/antoniopaya22/apollon"
    # with the adversarial attacks (ZOO, HopSkipJump)
    pip install "apollon-ids[attacks] @ git+https://github.com/antoniopaya22/apollon"
    ```

=== "uv"

    ```bash
    uv venv && source .venv/bin/activate
    uv pip install "apollon-ids[attacks] @ git+https://github.com/antoniopaya22/apollon"
    ```

## Desde el código fuente (desarrollo)

```bash
git clone https://github.com/antoniopaya22/apollon.git
cd apollon
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # package, attacks, pytest and ruff
pytest                       # runs in a few seconds on synthetic data
```

## Extras opcionales

| Extra | Instala | Necesario para |
|---|---|---|
| `attacks` | `adversarial-robustness-toolbox` | `apollon attack`, `apollon.attacks` |
| `docs` | `mkdocs-material`, `mkdocstrings` | generar este sitio (`mkdocs serve`) |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `attacks` | contribuir |

## Comprobar la instalación

```console
$ apollon --version
apollon 2.0.0
$ apollon demo
```

!!! note "Entorno de investigación original"
    Los notebooks de [`research/`](https://github.com/antoniopaya22/apollon/tree/main/research)
    se escribieron con las versiones fijadas en `research/requirements.txt` (Python 3.10,
    scikit-learn 1.2). El paquete en sí funciona con las versiones actuales.
