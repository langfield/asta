""" asta.constants """
import datetime
from typing import Dict, List
import torch
import numpy as np

# pylint: disable=invalid-name
NoneType = type(None)
EllipsisType = type(Ellipsis)
DIM_TYPES: List[type] = [int, EllipsisType, NoneType]  # type: ignore[misc]
ARRAY_TYPES: List[type] = [np.ndarray, torch.Tensor]
GENERIC_TYPES: List[type] = [
    bool,
    int,
    float,
    complex,
    bytes,
    str,
    datetime.datetime,
    datetime.timedelta,
]
NP_UNSIZED_TYPE_KINDS: Dict[type, str] = {bytes: "S", str: "U", object: "O"}
NP_GENERIC_TYPES: List[type] = [
    bool,
    int,
    float,
    complex,
    bytes,
    str,
    object,
    np.datetime64,
    np.timedelta64,
]
TORCH_GENERIC_TYPES: List[type] = [
    bool,
    int,
    float,
    bytes,
]
TORCH_DTYPE_MAP: Dict[type, torch.dtype] = {
    int: torch.int32,
    float: torch.float32,
    bool: torch.bool,
    bytes: torch.uint8,
}
