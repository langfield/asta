""" asta.constants """
import datetime
from typing import Dict, List, Any
import torch
import numpy as np

# pylint: disable=invalid-name, too-few-public-methods


# Metaclasses.
class ScalarMeta(type):
    """ A meta class for the ``Scalar`` class. """

    _GENERIC_TYPES: List[type]
    _ARRAY_TYPES: List[type]

    def __instancecheck__(cls, inst: Any) -> bool:
        """ Support expected behavior for ``isinstance(<number-like>, Scalar)``. """
        for arr_type in cls._ARRAY_TYPES:
            if isinstance(inst, arr_type):
                assert hasattr(inst, "shape")
                if inst.shape == tuple():  # type: ignore[attr-defined]
                    return True
                return False
        for generic_type in cls._GENERIC_TYPES:
            if isinstance(inst, generic_type):
                return True
        return False


# Types.
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
NoneType = type(None)
EllipsisType = type(Ellipsis)
DIM_TYPES: List[type] = [
    int,
    ScalarMeta,
    EllipsisType,
    NoneType,  # type: ignore[misc]
    tuple,
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
