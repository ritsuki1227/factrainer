from unittest.mock import MagicMock, patch

from factrainer.lightgbm.config import (
    LgbLearner,
    LgbModelConfig,
    LgbPredictor,
    LgbTrainConfig,
)


@patch("factrainer.lightgbm.config.LgbPredictor", spec=LgbPredictor)
@patch("factrainer.lightgbm.config.LgbLearner", spec=LgbLearner)
def test_create_lgb_model_config_with_default(
    learner: MagicMock, predictor: MagicMock
) -> None:
    train_config = MagicMock(spec=LgbTrainConfig)
    expected = LgbModelConfig(
        learner=learner.return_value,
        predictor=predictor.return_value,
        train_config=train_config,
        pred_config=None,
    )
    actual = LgbModelConfig.create(train_config)
    assert actual == expected
