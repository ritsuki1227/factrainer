from collections.abc import Sequence

from factrainer.base.raw_model import RawModel


class CvRawModels[U: RawModel](RawModel):
    models: Sequence[U]
