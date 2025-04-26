from pydantic import BaseModel, ConfigDict
from sklearn.base import BaseEstimator


class RawModel(BaseModel):
    estimator: BaseEstimator
    model_config = ConfigDict(arbitrary_types_allowed=True)
