from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import scipy
from factrainer.base.dataset import BaseDatasetSlicer, RowIndex

import lightgbm as lgb
from lightgbm.basic import (
    _LGBM_GroupType,
    _LGBM_InitScoreType,
    _LGBM_LabelType,
    _LGBM_PositionType,
    _LGBM_TrainDataType,
    _LGBM_WeightType,
)
from lightgbm.compat import (
    dt_DataTable,
)

from .types import (
    IsPaArray,
    IsPaChunkedArray,
    IsPaTable,
    IsPdDataFrame,
    IsPdSeries,
)


class LgbDataSlicer(BaseDatasetSlicer[_LGBM_TrainDataType]):
    def slice(self, data: _LGBM_TrainDataType, index: RowIndex) -> _LGBM_TrainDataType:
        if isinstance(data, str):
            raise NotImplementedError
        elif isinstance(data, Path):
            raise NotImplementedError
        elif isinstance(data, np.ndarray):
            return data[index]
        elif IsPdDataFrame().is_instance(data):
            return data.take(index)
        elif isinstance(data, dt_DataTable):
            raise NotImplementedError
        elif isinstance(data, scipy.sparse.spmatrix):
            raise NotImplementedError
        elif isinstance(data, Sequence):
            raise NotImplementedError
        elif isinstance(data, list):
            raise NotImplementedError
        elif IsPaTable().is_instance(data):
            raise NotImplementedError
        else:
            raise TypeError


class LgbLabelSlicer(BaseDatasetSlicer[_LGBM_LabelType]):
    def slice(self, data: _LGBM_LabelType, index: RowIndex) -> _LGBM_LabelType:
        if isinstance(data, list):
            raise NotImplementedError
        elif isinstance(data, np.ndarray):
            return data[index]
        elif IsPdDataFrame().is_instance(data):
            return data.take(index)
        elif IsPdSeries().is_instance(data):
            return data.take(index)
        elif isinstance(data, dt_DataTable):
            raise NotImplementedError
        elif IsPaArray().is_instance(data):
            raise NotImplementedError
        elif IsPaChunkedArray().is_instance(data):
            raise NotImplementedError
        else:
            raise TypeError


class LgbWeightSlicer(BaseDatasetSlicer[_LGBM_WeightType]):
    def slice(self, data: _LGBM_WeightType, index: RowIndex) -> _LGBM_WeightType:
        if isinstance(data, list):
            return [data[i] for i in index]
        elif isinstance(data, np.ndarray):
            return data[index]
        elif IsPdSeries().is_instance(data):
            return data.take(index)
        elif IsPaArray().is_instance(data):
            raise NotImplementedError
        elif IsPaChunkedArray().is_instance(data):
            raise NotImplementedError
        else:
            raise TypeError


class LgbInitScoreSlicer(BaseDatasetSlicer[_LGBM_InitScoreType]):
    def slice(self, data: _LGBM_InitScoreType, index: RowIndex) -> _LGBM_InitScoreType:
        if isinstance(data, list):
            raise NotImplementedError
        elif isinstance(data, np.ndarray):
            return data[index]
        elif IsPdDataFrame().is_instance(data):
            return data.take(index)
        elif IsPdSeries().is_instance(data):
            return data.take(index)
        elif IsPaTable().is_instance(data):
            raise NotImplementedError
        elif IsPaArray().is_instance(data):
            raise NotImplementedError
        elif IsPaChunkedArray().is_instance(data):
            raise NotImplementedError
        else:
            raise TypeError


class LgbGroupSlicer(BaseDatasetSlicer[_LGBM_GroupType]):
    def slice(self, data: _LGBM_GroupType, index: RowIndex) -> _LGBM_GroupType:
        if isinstance(data, list):
            raise NotImplementedError
        elif isinstance(data, np.ndarray):
            return data[index]
        elif IsPdSeries().is_instance(data):
            return data.take(index)
        elif IsPaArray().is_instance(data):
            raise NotImplementedError
        elif IsPaChunkedArray().is_instance(data):
            raise NotImplementedError
        else:
            raise TypeError


class LgbPositionSlicer(BaseDatasetSlicer[_LGBM_PositionType]):
    def slice(self, data: _LGBM_PositionType, index: RowIndex) -> _LGBM_PositionType:
        if isinstance(data, np.ndarray):
            return data[index]
        elif IsPdSeries().is_instance(data):
            return data.take(index)
        else:
            raise TypeError


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
