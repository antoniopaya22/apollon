"""Loading and preprocessing of the CIC intrusion detection datasets.

The pipeline follows the paper:

1. read and concatenate the CSV files of a dataset, shuffled with a fixed seed;
2. drop identifier columns (addresses, ports, timestamps) and constant features;
3. replace infinite and missing values by 0 and map labels to 0 (benign) / 1 (attack);
4. stratified train/test split (70/30);
5. scale with a ``RobustScaler`` (without centering) followed by a ``Normalizer``,
   both fitted on the training split only.

The result is a :class:`Dataset` that can be saved to and loaded from a single ``.npz``
file, which is what the command line interface works with.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, RobustScaler

__all__ = ["DATASETS", "Dataset", "DatasetSpec", "clean", "load_csvs", "prepare", "split_and_scale"]


@dataclass(frozen=True)
class DatasetSpec:
    """How to read one dataset."""

    name: str
    label: str
    benign: str
    drop: tuple[str, ...] = ()
    url: str = ""
    description: str = ""


#: Supported datasets, by command line name.
DATASETS: dict[str, DatasetSpec] = {
    "cic-ids2017": DatasetSpec(
        name="CIC-IDS2017",
        label=" Label",
        benign="BENIGN",
        url="https://www.unb.ca/cic/datasets/ids-2017.html",
        description="Five days of traffic with brute force, DoS, DDoS, web attacks, "
        "infiltration, botnet and port scans (MachineLearningCSV files).",
    ),
    "cse-cic-ids2018": DatasetSpec(
        name="CSE-CIC-IDS2018",
        label="Label",
        benign="Benign",
        drop=("Timestamp",),
        url="https://www.unb.ca/cic/datasets/ids-2018.html",
        description="Ten days of traffic of a larger network on AWS with the same attack "
        "families as CIC-IDS2017.",
    ),
    "cic-ddos2019": DatasetSpec(
        name="CIC-DDoS2019",
        label=" Label",
        benign="BENIGN",
        drop=(
            "Unnamed: 0",
            "Flow ID",
            " Source IP",
            " Source Port",
            " Destination IP",
            " Destination Port",
            " Timestamp",
            " Inbound",
            "SimillarHTTP",
            " Protocol",
        ),
        url="https://www.unb.ca/cic/datasets/ddos-2019.html",
        description="Reflection and exploitation DDoS attacks (DNS, LDAP, MSSQL, NetBIOS, "
        "NTP, SNMP, SSDP, UDP, SYN, TFTP, UDP-lag).",
    ),
}


@dataclass
class Dataset:
    """A preprocessed train/test split."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    name: str = ""

    def save(self, path: str | Path) -> Path:
        """Save to a compressed ``.npz`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X_train=self.X_train,
            X_test=self.X_test,
            y_train=self.y_train,
            y_test=self.y_test,
            feature_names=np.array(self.feature_names, dtype=str),
            name=np.array(self.name),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        """Load a dataset saved with :meth:`save`."""
        with np.load(path, allow_pickle=False) as f:
            return cls(
                f["X_train"],
                f["X_test"],
                f["y_train"],
                f["y_test"],
                [str(c) for c in f["feature_names"]],
                str(f["name"]),
            )

    def summary(self) -> str:
        n_tr, n_te = len(self.y_train), len(self.y_test)
        return (
            f"{self.name or 'dataset'}: {self.X_train.shape[1]} features, "
            f"{n_tr} train ({self.y_train.mean():.1%} attacks), "
            f"{n_te} test ({self.y_test.mean():.1%} attacks)"
        )


def load_csvs(paths: Iterable[str | Path], seed: int = 0) -> pd.DataFrame:
    """Read and concatenate CSV files (a directory is expanded to its ``*.csv`` files)."""
    files: list[Path] = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.csv")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError("no CSV files found")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def clean(
    df: pd.DataFrame, spec: DatasetSpec, drop_duplicates: bool = False
) -> tuple[pd.DataFrame, np.ndarray]:
    """Features and binary labels (0 benign, 1 attack) of a raw dataframe."""
    if spec.label not in df.columns:
        raise KeyError(f"label column {spec.label!r} not found; is this {spec.name}?")
    df = df[df[spec.label] != spec.label]  # header lines repeated inside some CSVs
    df = df.drop(columns=[c for c in spec.drop if c in df.columns])
    if drop_duplicates:
        df = df.drop_duplicates()
    y = (df[spec.label].astype(str).str.strip() != spec.benign).to_numpy(dtype=np.int8)
    X = df.drop(columns=[spec.label]).apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.nunique(dropna=False) > 1]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return X, y


def split_and_scale(
    X: pd.DataFrame | np.ndarray,
    y,
    test_size: float = 0.3,
    seed: int = 0,
    name: str = "",
    normalize: bool = True,
) -> Dataset:
    """Stratified split, then ``RobustScaler`` (no centering) and, with ``normalize``, a
    ``Normalizer`` that scales every row to unit norm (as in the paper)."""
    names = [str(c) for c in X.columns] if isinstance(X, pd.DataFrame) else []
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(np.int8)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = RobustScaler(with_centering=False)
    X_tr = np.nan_to_num(scaler.fit_transform(X_tr))
    X_te = np.nan_to_num(scaler.transform(X_te))
    if normalize:
        norm = Normalizer().fit(X_tr)
        X_tr, X_te = norm.transform(X_tr), norm.transform(X_te)
    return Dataset(X_tr.astype(np.float32), X_te.astype(np.float32), y_tr, y_te, names, name)


def prepare(
    dataset: str,
    paths: Iterable[str | Path],
    test_size: float = 0.3,
    seed: int = 0,
    sample: int | None = None,
    drop_duplicates: bool = False,
) -> Dataset:
    """Full pipeline for one of :data:`DATASETS`, from raw CSV files to a :class:`Dataset`.

    ``sample`` keeps a stratified random subset of that many rows, which is handy for
    quick experiments on the larger datasets.
    """
    try:
        spec = DATASETS[dataset]
    except KeyError:
        raise ValueError(f"unknown dataset {dataset!r}; available: {', '.join(DATASETS)}") from None
    X, y = clean(load_csvs(paths, seed), spec, drop_duplicates)
    if sample is not None and sample < len(y):
        X, _, y, _ = train_test_split(X, y, train_size=sample, random_state=seed, stratify=y)
    return split_and_scale(X, y, test_size, seed, spec.name)
