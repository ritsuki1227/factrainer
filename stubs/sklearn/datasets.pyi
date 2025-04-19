from typing import Optional, Union

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
