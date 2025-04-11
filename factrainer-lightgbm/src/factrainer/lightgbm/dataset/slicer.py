from collections.abc import Sequence
from pathlib import Path

import numpy as np
import scipy
from factrainer.base.dataset import BaseDatasetSlicer, RowIndex

import lightgbm as lgb
from lightgbm.compat import (  # type: ignore
    dt_DataTable,
    pa_Table,
    pd_DataFrame,
    pd_Series,
)

from .type import (
    LgbDataType,
    LgbGroupType,
    LgbInitScoreType,
    LgbLabelType,
    LgbPositionType,
    LgbWeightType,
)


class LgbDataSlicer(BaseDatasetSlicer[LgbDataType]):
    def slice(self, data: LgbDataType, index: RowIndex) -> LgbDataType:
        match data:
            case str():
                raise NotImplementedError
            case Path():
                raise NotImplementedError
            case np.ndarray():
                return data[index]
            case pd_DataFrame():
                return data.iloc[index]  # type: ignore
            case dt_DataTable():
                raise NotImplementedError
            case scipy.sparse.spmatrix():
                raise NotImplementedError
            case Sequence():
                raise NotImplementedError
            case pa_Table():
                raise NotImplementedError
            case _:
                raise NotImplementedError


class LgbLabelSlicer(BaseDatasetSlicer[LgbLabelType]):
    def slice(self, data: LgbLabelType, index: RowIndex) -> LgbLabelType:
        match data:
            case list():
                raise NotImplementedError
            case np.ndarray():
                return data[index]
            case pd_DataFrame():
                return data.iloc[index]
            case pd_Series():
                return data.iloc[index]
            case dt_DataTable():
                raise NotImplementedError
            case scipy.sparse.spmatrix():
                raise NotImplementedError
            case Sequence():
                raise NotImplementedError
            case pa_Table():
                raise NotImplementedError
            case _:
                raise NotImplementedError


class LgbWeightSlicer(BaseDatasetSlicer[LgbWeightType]):
    def slice(self, data: LgbWeightType, index: RowIndex) -> LgbWeightType:
        raise NotImplementedError


class LgbInitScoreSlicer(BaseDatasetSlicer[LgbInitScoreType]):
    def slice(self, data: LgbInitScoreType, index: RowIndex) -> LgbInitScoreType:
        raise NotImplementedError


class LgbGroupSlicer(BaseDatasetSlicer[LgbGroupType]):
    def slice(self, data: LgbGroupType, index: RowIndex) -> LgbGroupType:
        raise NotImplementedError


class LgbPositionSlicer(BaseDatasetSlicer[LgbPositionType]):
    def slice(self, data: LgbPositionType, index: RowIndex) -> LgbPositionType:
        raise NotImplementedError


class LgbDatasetSlicer(BaseDatasetSlicer[lgb.Dataset]):
    def __init__(self, reference: lgb.Dataset | None = None) -> None:
        self.reference = reference

    def slice(self, data: lgb.Dataset, index: RowIndex) -> lgb.Dataset:
        return lgb.Dataset(
            data=LgbDataSlicer().slice(data.data, index),
            label=(
                LgbLabelSlicer().slice(data.label, index)
                if data.label is not None
                else None
            ),
            reference=self.reference,
            weight=(
                LgbWeightSlicer().slice(data.weight, index)
                if data.weight is not None
                else None
            ),
            group=(
                LgbGroupSlicer().slice(data.group, index)
                if data.group is not None
                else None
            ),
            init_score=(
                LgbInitScoreSlicer().slice(data.init_score, index)
                if data.init_score is not None
                else None
            ),
            feature_name=data.feature_name,
            categorical_feature=data.categorical_feature,
            params=data.params,
            free_raw_data=data.free_raw_data,
            position=(
                LgbPositionSlicer().slice(data.position, index)
                if data.position is not None
                else None
            ),
        )
