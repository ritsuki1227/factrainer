from collections.abc import Generator

import numpy as np
from factrainer.base.dataset import IndexableDataset, RowIndex, Rows
from sklearn.model_selection._split import _BaseKFold

import lightgbm as lgb

from .slicer import LgbDatasetSlicer
from .types import IsPdDataFrame


class LgbDataset(IndexableDataset):
    """Wrapper for LightGBM Dataset providing factrainer-compatible interface.

    This class wraps a native `lgb.Dataset` to provide a consistent interface
    for cross-validation and data manipulation within the factrainer framework.
    This dataset is passed to the `train` and `predict` methods of
    `SingleModelContainer` and `CvModelContainer` when using LightGBM models.

    Attributes
    ----------
    dataset : lgb.Dataset
        The underlying LightGBM Dataset instance containing features, labels,
        and optional metadata like weights, groups, and init scores.

    Examples
    --------
    >>> import numpy as np
    >>> import lightgbm as lgb
    >>> from factrainer.lightgbm import LgbDataset
    >>> # Create from numpy arrays
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randn(100)
    >>> lgb_data = lgb.Dataset(X, label=y)
    >>> dataset = LgbDataset(dataset=lgb_data)
    >>> # Create with additional metadata
    >>> weights = np.random.rand(100)
    >>> lgb_data = lgb.Dataset(X, label=y, weight=weights)
    >>> dataset = LgbDataset(dataset=lgb_data)
    """

    dataset: lgb.Dataset

    def k_fold_split(
        self, k_fold: _BaseKFold
    ) -> Generator[tuple[RowIndex, RowIndex], None, None]:
        if isinstance(self.dataset.data, np.ndarray) or IsPdDataFrame().is_instance(
            self.dataset.data
        ):
            for train_index, val_index in k_fold.split(self.dataset.data):
                yield train_index, val_index
        else:
            raise NotImplementedError

    def __getitem__(self, index: Rows) -> "LgbDataset":
        match index:
            case int():
                return LgbDataset(
                    dataset=LgbDatasetSlicer(self.dataset.reference).slice(
                        self.dataset, [index]
                    )
                )
            case list():
                return LgbDataset(
                    dataset=LgbDatasetSlicer(self.dataset.reference).slice(
                        self.dataset, index
                    )
                )
            case np.ndarray():
                return LgbDataset(
                    dataset=LgbDatasetSlicer(self.dataset.reference).slice(
                        self.dataset, index
                    )
                )
            case slice():
                raise NotImplementedError
            case _:
                raise TypeError
