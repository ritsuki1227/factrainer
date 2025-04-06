from ._split import _BaseKFold

class _UnsupportedGroupCVMixin: ...

class KFold(_UnsupportedGroupCVMixin, _BaseKFold):
    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None: ...
