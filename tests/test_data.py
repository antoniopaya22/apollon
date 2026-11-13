import numpy as np
import pandas as pd
import pytest

from apollon.data import DATASETS, Dataset, clean, load_csvs, prepare, split_and_scale


def raw_cic17(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            " Destination Port": rng.integers(1, 65535, n),
            " Flow Duration": rng.exponential(1e5, n),
            " Total Fwd Packets": rng.integers(1, 50, n),
            "Flow Bytes/s": rng.exponential(1e3, n),
            " Constant": np.ones(n),
            " Label": rng.choice(["BENIGN", "DoS Hulk", "PortScan"], n, p=[0.7, 0.2, 0.1]),
        }
    )
    df.loc[0, "Flow Bytes/s"] = np.inf
    return df


def test_clean_binarises_labels_and_drops_constants():
    df = raw_cic17()
    X, y = clean(df, DATASETS["cic-ids2017"])
    assert " Constant" not in X.columns and " Label" not in X.columns
    assert set(np.unique(y)) == {0, 1}
    assert (y == 0).sum() == (df[" Label"] == "BENIGN").sum()
    assert np.isfinite(X.to_numpy()).all()


def test_clean_rejects_wrong_dataset():
    with pytest.raises(KeyError, match="label column"):
        clean(raw_cic17(), DATASETS["cse-cic-ids2018"])


def test_prepare_from_directory_and_roundtrip(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    df = raw_cic17(600)
    df.iloc[:300].to_csv(raw / "monday.csv", index=False)
    df.iloc[300:].to_csv(raw / "friday.csv", index=False)
    assert len(load_csvs([raw])) == 600
    ds = prepare("cic-ids2017", [raw], seed=1)
    assert len(ds.y_train) + len(ds.y_test) == 600
    assert abs(ds.y_train.mean() - ds.y_test.mean()) < 0.05  # stratified
    np.testing.assert_allclose(np.linalg.norm(ds.X_train, axis=1), 1, atol=1e-5)
    back = Dataset.load(ds.save(tmp_path / "d.npz"))
    np.testing.assert_array_equal(back.X_test, ds.X_test)
    assert back.feature_names == ds.feature_names and back.name == "CIC-IDS2017"


def test_split_without_normalisation():
    X = np.random.default_rng(0).random((200, 4))
    y = np.r_[np.zeros(150), np.ones(50)]
    ds = split_and_scale(X, y, normalize=False)
    assert not np.allclose(np.linalg.norm(ds.X_train, axis=1), 1)


def test_unknown_dataset(tmp_path):
    with pytest.raises(ValueError, match="unknown dataset"):
        prepare("kdd99", [tmp_path])
