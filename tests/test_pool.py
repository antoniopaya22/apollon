import pytest

from apollon import ARMS, DEFAULT_POOL, default_arms, make_arm, make_pool


def test_default_pool_is_the_paper_pool():
    assert DEFAULT_POOL == ("RF", "DT", "NB", "LR", "MLP")
    assert [type(a).__name__ for a in default_arms(0)] == [
        "RandomForestClassifier",
        "DecisionTreeClassifier",
        "GaussianNB",
        "LogisticRegression",
        "MLPClassifier",
    ]


@pytest.mark.parametrize("name", sorted(ARMS))
def test_every_arm_can_be_built(name):
    arm = make_arm(name.lower(), 0)
    assert hasattr(arm, "fit") and hasattr(arm, "predict_proba")


def test_unknown_arm():
    with pytest.raises(ValueError, match="unknown arm"):
        make_pool(["RF", "XGB"])
