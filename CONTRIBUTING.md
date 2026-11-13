# Contributing to Apollon

Thanks for your interest in Apollon. Bug reports, questions, documentation fixes and new
features are all welcome.

## Reporting issues

Open an [issue](https://github.com/antoniopaya22/apollon/issues) with the version
(`apollon --version`), your Python and scikit-learn versions, what you ran and what you
expected. A minimal example on synthetic data (`sklearn.datasets.make_classification`)
helps a lot.

## Development setup

```bash
git clone https://github.com/antoniopaya22/apollon.git
cd apollon
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"
```

## Before opening a pull request

```bash
ruff check src tests          # lint
ruff format src tests         # format
pytest --cov                  # tests (a few seconds, synthetic data only)
mkdocs build --strict         # documentation, if you changed it
```

- Keep pull requests focused on one change and describe the motivation.
- Add tests for new behaviour; tests must not need the datasets.
- Document public functions with NumPy-style docstrings: they become the
  [API reference](https://antoniopaya22.github.io/apollon/reference/).
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages
  (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).
- Add an entry to `CHANGELOG.md` under *Unreleased*.

## Research code

`research/` keeps the notebooks and scripts that produced the paper. Changes there
should preserve what they reproduce; new functionality belongs in `src/apollon/`.

## Code of conduct

This project follows the [Contributor Covenant](https://github.com/antoniopaya22/apollon/blob/main/CODE_OF_CONDUCT.md). Be respectful and
constructive.
