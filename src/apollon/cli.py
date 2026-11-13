"""Command line interface: ``apollon <command> --help``.

apollon prepare  --dataset cic-ids2017 --raw data/CIC-IDS2017 --out data/cic17.npz
apollon train    --data data/cic17.npz --out models/apollon.joblib
apollon evaluate --data data/cic17.npz --model models/apollon.joblib
apollon attack   --data data/cic17.npz --model models/apollon.joblib --attack hsja
apollon demo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from apollon import __version__
from apollon.data import DATASETS, Dataset, prepare, split_and_scale
from apollon.mab import ApollonMAB, load_model, save_model
from apollon.metrics import evaluate, format_metrics
from apollon.pool import ARMS, DEFAULT_POOL, make_arm, make_pool


def _print_metrics(title: str, m: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"title": title, **m}))
    else:
        print(f"\n{title}\n{'-' * len(title)}\n{format_metrics(m)}")


def cmd_prepare(a) -> None:
    ds = prepare(
        a.dataset,
        a.raw,
        test_size=a.test_size,
        seed=a.seed,
        sample=a.sample,
        drop_duplicates=a.drop_duplicates,
    )
    ds.save(a.out)
    print(ds.summary())
    print(f"saved to {a.out}")


def cmd_train(a) -> None:
    ds = Dataset.load(a.data)
    print(ds.summary())
    model = ApollonMAB(
        arms=make_pool(a.arms, a.seed),
        n_clusters=a.clusters,
        val_size=a.val_size,
        prior_strength=a.prior_strength,
        random_state=a.seed,
    )
    t = time.perf_counter()
    model.fit(ds.X_train, ds.y_train)
    print(f"trained {len(a.arms)} arms x {a.clusters} clusters in {time.perf_counter() - t:.1f} s")
    save_model(model, a.out)
    print(f"saved to {a.out}")


def cmd_evaluate(a) -> None:
    ds, model = Dataset.load(a.data), load_model(a.model)
    pred, arms = model.predict(ds.X_test, return_arms=True)
    _print_metrics("Apollon", evaluate(ds.y_test, pred), a.json)
    names = list(a.arm_names or DEFAULT_POOL)[: model.alpha_.shape[1]]
    share = np.bincount(arms, minlength=model.alpha_.shape[1]) / len(arms)
    if not a.json:
        print("\nArm selection")
        for n, s in zip(names, share, strict=False):
            print(f"  {n:<5} {s:6.1%}")


def cmd_attack(a) -> None:
    from apollon.attacks import (
        apollon_classifier,
        functional_indexes,
        model_classifier,
        restore_functional,
        run_attack,
    )

    ds, model = Dataset.load(a.data), load_model(a.model)
    rng = np.random.default_rng(a.seed)
    atk = np.flatnonzero(ds.y_test == 1)
    idx = rng.choice(atk, size=min(a.n, len(atk)), replace=False)
    x = ds.X_test[idx]
    if a.setting == "direct":
        clf = apollon_classifier(model, ds.X_test.shape[1])
    else:
        surrogate = make_arm(a.surrogate, a.seed).fit(ds.X_train, ds.y_train)
        clf = model_classifier(surrogate)
    t = time.perf_counter()
    adv = run_attack(a.attack, clf, x, max_iter=a.max_iter)
    func, _, _ = functional_indexes(ds.feature_names)
    if func and not a.no_constraints:
        adv = restore_functional(adv, x, func)
    elapsed = time.perf_counter() - t
    before = evaluate(np.ones(len(x)), model.predict(x))["detection_rate"]
    after = evaluate(np.ones(len(adv)), model.predict(adv))["detection_rate"]
    m = {
        "detection_rate_clean": before,
        "detection_rate_adversarial": after,
        "evasion_rate": 1 - after,
        "samples": float(len(x)),
        "seconds": elapsed,
    }
    _print_metrics(f"{a.attack.upper()} ({a.setting}) against Apollon", m, a.json)


def cmd_demo(a) -> None:
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=a.samples,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        class_sep=1.5,
        flip_y=0.01,
        weights=[0.8, 0.2],
        random_state=a.seed,
    )
    ds = split_and_scale(X, y, seed=a.seed, name="synthetic", normalize=False)
    if not a.json:
        print(ds.summary())
    model = ApollonMAB(random_state=a.seed).fit(ds.X_train, ds.y_train)
    pred, arms = model.predict(ds.X_test, return_arms=True)
    _print_metrics("Apollon on synthetic traffic", evaluate(ds.y_test, pred), a.json)
    if not a.json:
        share = np.bincount(arms, minlength=len(DEFAULT_POOL)) / len(arms)
        mix = ", ".join(f"{n} {s:.0%}" for n, s in zip(DEFAULT_POOL, share, strict=False))
        print(f"\nArm selection: {mix}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="apollon",
        description="Apollon: a robust defence "
        "system against adversarial machine learning attacks in IDS.",
    )
    p.add_argument("--version", action="version", version=f"apollon {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("prepare", help="preprocess a dataset into a .npz split")
    s.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    s.add_argument("--raw", nargs="+", required=True, help="CSV files or directories")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--test-size", type=float, default=0.3)
    s.add_argument("--sample", type=int, default=None, help="keep a stratified subset of N rows")
    s.add_argument("--drop-duplicates", action="store_true")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("train", help="train Apollon on a prepared dataset")
    s.add_argument("--data", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument(
        "--arms",
        nargs="+",
        default=list(DEFAULT_POOL),
        choices=sorted(ARMS),
        metavar="ARM",
        help=f"classifier pool (default: {' '.join(DEFAULT_POOL)})",
    )
    s.add_argument("--clusters", type=int, default=2)
    s.add_argument("--val-size", type=float, default=0.2)
    s.add_argument("--prior-strength", type=float, default=None)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("evaluate", help="detection metrics on the test split")
    s.add_argument("--data", type=Path, required=True)
    s.add_argument("--model", type=Path, required=True)
    s.add_argument("--arm-names", nargs="+", default=None)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("attack", help="adversarial evaluation with ZOO or HopSkipJump")
    s.add_argument("--data", type=Path, required=True)
    s.add_argument("--model", type=Path, required=True)
    s.add_argument("--attack", choices=["zoo", "hsja"], default="hsja")
    s.add_argument(
        "--setting",
        choices=["direct", "transfer"],
        default="direct",
        help="query Apollon itself, or craft against a surrogate and transfer",
    )
    s.add_argument("--surrogate", default="RF", choices=sorted(ARMS))
    s.add_argument("--n", type=int, default=100, help="attack samples")
    s.add_argument("--max-iter", type=int, default=100)
    s.add_argument(
        "--no-constraints",
        action="store_true",
        help="do not restore functional features after the attack",
    )
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_attack)

    s = sub.add_parser("demo", help="train and evaluate on synthetic data (no download)")
    s.add_argument("--samples", type=int, default=5000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
