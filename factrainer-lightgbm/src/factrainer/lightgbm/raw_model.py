from factrainer.base.raw_model import RawModel

import lightgbm as lgb


class LgbModel(RawModel):
    model: lgb.Booster
