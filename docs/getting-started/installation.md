# Installation

Apollon needs **Python 3.10 or newer**. It depends only on NumPy, pandas, scikit-learn
and joblib; the adversarial evaluation adds the
[Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox).

## From GitHub

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

## From source (development)

```bash
git clone https://github.com/antoniopaya22/apollon.git
cd apollon
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # package, attacks, pytest and ruff
pytest                       # runs in a few seconds on synthetic data
```

## Optional extras

| Extra | Installs | Needed for |
|---|---|---|
| `attacks` | `adversarial-robustness-toolbox` | `apollon attack`, `apollon.attacks` |
| `docs` | `mkdocs-material`, `mkdocstrings` | building this site (`mkdocs serve`) |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `attacks` | contributing |

## Check the installation

```console
$ apollon --version
apollon 2.0.0
$ apollon demo
```

!!! note "Original research environment"
    The notebooks in [`research/`](https://github.com/antoniopaya22/apollon/tree/main/research)
    were written with the pinned versions of `research/requirements.txt` (Python 3.10,
    scikit-learn 1.2). The package itself works with current releases.
