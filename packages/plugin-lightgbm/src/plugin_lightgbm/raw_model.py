import lightgbm as lgb
from base.raw_model import RawModel


class LgbModel(RawModel):
    model: lgb.Booster
