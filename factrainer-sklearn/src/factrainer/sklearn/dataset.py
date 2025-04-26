from collections.abc import Generator

from factrainer.base.dataset import IndexableDataset, RowIndex, RowsAndColumns

from sklearn.model_selection._split import _BaseKFold


class SklearnDataset(IndexableDataset):
    def k_fold_split(
        self, k_fold: _BaseKFold
    ) -> Generator[tuple[RowIndex, RowIndex], None, None]:
        raise NotImplementedError

    def __getitem__(self, index: RowsAndColumns) -> "SklearnDataset":
        raise NotImplementedError
