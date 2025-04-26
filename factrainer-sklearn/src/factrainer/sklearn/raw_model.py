from factrainer.base.raw_model import RawModel

from sklearn.base import BaseEstimator


class SklearnModel(RawModel):
    estimator: BaseEstimator
