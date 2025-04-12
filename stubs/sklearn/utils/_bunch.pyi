from typing import Any

import numpy as np
from numpy import typing as npt

class Bunch:
    data: npt.NDArray[np.number[Any]] = ...
    target: npt.NDArray[np.number[Any]] = ...
