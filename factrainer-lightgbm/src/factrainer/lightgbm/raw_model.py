from factrainer.base.raw_model import RawModel

import lightgbm as lgb


class LgbModel(RawModel):
    """Wrapper for trained LightGBM model.

    This class wraps a trained `lgb.Booster` instance to provide
    a consistent interface within the factrainer framework.

    Attributes
    ----------
    model : lgb.Booster
        The trained LightGBM Booster instance.
    """

    model: lgb.Booster
