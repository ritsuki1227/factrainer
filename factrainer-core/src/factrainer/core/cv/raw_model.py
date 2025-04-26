from collections.abc import Sequence

from factrainer.base.raw_model import RawModel


class RawModels[U: RawModel](RawModel):
    models: Sequence[U]
