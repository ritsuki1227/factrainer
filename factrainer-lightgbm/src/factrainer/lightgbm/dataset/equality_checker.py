from collections.abc import Sequence
from pathlib import Path

import numpy as np
import scipy
from factrainer.base.dataset import BaseDatasetEqualityChecker

import lightgbm as lgb
from lightgbm.basic import (
    _LGBM_CategoricalFeatureConfiguration,
    _LGBM_FeatureNameConfiguration,
    _LGBM_GroupType,
    _LGBM_InitScoreType,
    _LGBM_LabelType,
    _LGBM_PositionType,
    _LGBM_WeightType,
)
from lightgbm.compat import (  # type: ignore
    dt_DataTable,
    pa_Array,
    pa_ChunkedArray,
    pa_Table,
    pd_DataFrame,
    pd_Series,
)

from .type import LgbDataType


class LgbDataEqualityChecker(BaseDatasetEqualityChecker[LgbDataType]):
    def check(self, left: LgbDataType, right: LgbDataType) -> bool:
        match left:
            case str():
                raise NotImplementedError
            case Path():
                raise NotImplementedError
            case np.ndarray():
                return (
                    np.array_equal(left, right) if type(right) is np.ndarray else False
                )
            case pd_DataFrame():
                return left.equals(right) if type(right) is pd_DataFrame else False
            case dt_DataTable():
                raise NotImplementedError
            case scipy.sparse.spmatrix():
                raise NotImplementedError
            case list():
                raise NotImplementedError
            case Sequence():
                raise NotImplementedError
            case pa_Table():
                raise NotImplementedError
            case _:
                raise TypeError("Invalid type")


class LgbLabelEqualityChecker(BaseDatasetEqualityChecker[_LGBM_LabelType | None]):
    def check(
        self, left: _LGBM_LabelType | None, right: _LGBM_LabelType | None
    ) -> bool:
        match left:
            case list():
                return left == right if type(right) is list else False
            case np.ndarray():
                return (
                    np.array_equal(left, right) if type(right) is np.ndarray else False
                )
            case pd_Series():
                return left.equals(right) if type(right) is pd_Series else False
            case pd_DataFrame():
                return left.equals(right) if type(right) is pd_DataFrame else False
            case pa_Array():
                raise NotImplementedError
            case pa_ChunkedArray():
                raise NotImplementedError
            case None:
                return right is None
            case _:
                raise TypeError("Invalid type")


class LgbWeightEqualityChecker(BaseDatasetEqualityChecker[_LGBM_WeightType | None]):
    def check(
        self, left: _LGBM_WeightType | None, right: _LGBM_WeightType | None
    ) -> bool:
        match left:
            case np.ndarray():
                return (
                    np.array_equal(left, right) if type(right) is np.ndarray else False
                )
            case pd_Series():
                return left.equals(right) if type(right) is pd_Series else False
            case None:
                return right is None
            case _:
                raise TypeError("Invalid type")


class LgbInitScoreEqualityChecker(
    BaseDatasetEqualityChecker[_LGBM_InitScoreType | None]
):
    def check(
        self, left: _LGBM_InitScoreType | None, right: _LGBM_InitScoreType | None
    ) -> bool:
        match left:
            case list():
                raise NotImplementedError
            case np.ndarray():
                return (
                    np.array_equal(left, right) if type(right) is np.ndarray else False
                )
            case pd_DataFrame():
                return left.equals(right) if type(right) is pd_DataFrame else False
            case pd_Series():
                return left.equals(right) if type(right) is pd_Series else False
            case pa_Table():
                raise NotImplementedError
            case pa_Array():
                raise NotImplementedError
            case pa_ChunkedArray():
                raise NotImplementedError
            case None:
                return right is None
            case _:
                raise TypeError("Invalid type")


class LgbGroupEqualityChecker(BaseDatasetEqualityChecker[_LGBM_GroupType | None]):
    def check(
        self, left: _LGBM_GroupType | None, right: _LGBM_GroupType | None
    ) -> bool:
        match left:
            case list():
                raise NotImplementedError
            case np.ndarray():
                return (
                    np.array_equal(left, right) if type(right) is np.ndarray else False
                )
            case pd_Series():
                return left.equals(right) if type(right) is pd_Series else False
            case pa_Array():
                raise NotImplementedError
            case pa_ChunkedArray():
                raise NotImplementedError
            case None:
                return right is None
            case _:
                raise TypeError("Invalid type")


class LgbFeatureNameEqualityChecker(
    BaseDatasetEqualityChecker[_LGBM_FeatureNameConfiguration]
):
    def check(
        self,
        left: _LGBM_FeatureNameConfiguration,
        right: _LGBM_FeatureNameConfiguration,
    ) -> bool:
        match left:
            case list() | "auto":
                return left == right
            case _:
                raise TypeError("Invalid type")


class LgbCategoricalFeatureEqualityChecker(
    BaseDatasetEqualityChecker[_LGBM_CategoricalFeatureConfiguration]
):
    def check(
        self,
        left: _LGBM_CategoricalFeatureConfiguration,
        right: _LGBM_CategoricalFeatureConfiguration,
    ) -> bool:
        return left == right


class LgbPositionEqualityChecker(BaseDatasetEqualityChecker[_LGBM_PositionType | None]):
    def check(
        self, left: _LGBM_PositionType | None, right: _LGBM_PositionType | None
    ) -> bool:
        return left == right


class LgbDatasetEqualityChecker(BaseDatasetEqualityChecker[lgb.Dataset | None]):
    def check(self, left: lgb.Dataset | None, right: lgb.Dataset | None) -> bool:
        if left is None:
            return right is None
        if type(right) is not lgb.Dataset:
            return False
        return all(
            [
                LgbDataEqualityChecker().check(left.data, right.data),
                LgbLabelEqualityChecker().check(left.label, right.label),
                LgbDatasetEqualityChecker().check(left.reference, right.reference),
                LgbWeightEqualityChecker().check(left.weight, right.weight),
                LgbGroupEqualityChecker().check(left.group, right.group),
                LgbInitScoreEqualityChecker().check(left.init_score, right.init_score),
                LgbFeatureNameEqualityChecker().check(
                    left.feature_name, right.feature_name
                ),
                LgbCategoricalFeatureEqualityChecker().check(
                    left.categorical_feature, right.categorical_feature
                ),
                left.params == right.params,
                left.free_raw_data == right.free_raw_data,
                LgbPositionEqualityChecker().check(left.position, right.position),
            ]
        )
