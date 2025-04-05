from pydantic import BaseModel, ConfigDict


class RawModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
