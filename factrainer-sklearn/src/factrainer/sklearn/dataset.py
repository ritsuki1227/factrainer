from collections.abc import Generator

from factrainer.base.dataset import IndexableDataset, RowIndex, RowsAndColumns
from numpy import typing as npt

from sklearn.model_selection._split import _BaseKFold


class SklearnDataset(IndexableDataset):
    X: npt.ArrayLike
    y: npt.ArrayLike

    def k_fold_split(
        self, k_fold: _BaseKFold
    ) -> Generator[tuple[RowIndex, RowIndex], None, None]:
        raise NotImplementedError

    def __getitem__(self, index: RowsAndColumns) -> "SklearnDataset":
        raise NotImplementedError
