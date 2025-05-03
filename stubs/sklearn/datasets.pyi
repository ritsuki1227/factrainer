from typing import Any, Literal, Optional, Union, overload

from numpy import typing as npt

class Bunch: ...

def fetch_california_housing(
    *,
    data_home: Optional[str] = None,
    download_if_missing: bool = True,
    return_X_y: bool = False,
    as_frame: bool = False,
    n_retries: int = 3,
    delay: float = 1.0,
) -> Bunch: ...
def fetch_openml(
    *,
    name: Optional[str] = None,
    version: Union[str, int] = "active",
    data_id: Optional[int] = None,
    data_home: Optional[str] = None,
    target_column: Optional[Union[str, list[str]]] = "default-target",
    cache: bool = True,
    return_X_y: bool = False,
    as_frame: bool = False,
    parser: str = "auto",
    n_retries: int = 3,
    delay: float = 1.0,
) -> Bunch: ...
@overload
def load_iris(
    *,
    return_X_y: Literal[False],
    as_frame: bool = False,
) -> Bunch: ...
@overload
def load_iris(
    *,
    return_X_y: Literal[True],
    as_frame: bool = False,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]: ...
@overload
def load_iris(
    *,
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch | tuple[npt.NDArray[Any], npt.NDArray[Any]]: ...
@overload
def load_breast_cancer(
    *,
    return_X_y: Literal[False],
    as_frame: bool = False,
) -> Bunch: ...
@overload
def load_breast_cancer(
    *,
    return_X_y: Literal[True],
    as_frame: bool = False,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]: ...
@overload
def load_breast_cancer(
    *,
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch | tuple[npt.NDArray[Any], npt.NDArray[Any]]: ...
