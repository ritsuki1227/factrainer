from factrainer.core import hello
from factrainer.xgboost import hello as hello2


def test_hello() -> None:
    hello()
    hello2()
