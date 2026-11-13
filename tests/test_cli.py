import json

import pytest

from apollon.cli import main
from tests.test_data import raw_cic17


def test_demo(capsys):
    assert main(["demo", "--samples", "800", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["accuracy"] > 0.8


def test_prepare_train_evaluate(tmp_path, capsys):
    raw_cic17(600).to_csv(tmp_path / "day.csv", index=False)
    data, model = tmp_path / "d.npz", tmp_path / "m.joblib"
    main(
        [
            "prepare",
            "--dataset",
            "cic-ids2017",
            "--raw",
            str(tmp_path / "day.csv"),
            "--out",
            str(data),
        ]
    )
    main(["train", "--data", str(data), "--out", str(model), "--arms", "DT", "NB"])
    capsys.readouterr()
    main(["evaluate", "--data", str(data), "--model", str(model), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert 0 <= out["detection_rate"] <= 1


def test_attack(tmp_path, capsys):
    pytest.importorskip("art")
    raw_cic17(600).to_csv(tmp_path / "day.csv", index=False)
    data, model = tmp_path / "d.npz", tmp_path / "m.joblib"
    main(
        [
            "prepare",
            "--dataset",
            "cic-ids2017",
            "--raw",
            str(tmp_path / "day.csv"),
            "--out",
            str(data),
        ]
    )
    main(["train", "--data", str(data), "--out", str(model), "--arms", "DT", "NB"])
    capsys.readouterr()
    main(
        [
            "attack",
            "--data",
            str(data),
            "--model",
            str(model),
            "--attack",
            "hsja",
            "--n",
            "3",
            "--max-iter",
            "1",
            "--json",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert out["samples"] == 3 and 0 <= out["evasion_rate"] <= 1
