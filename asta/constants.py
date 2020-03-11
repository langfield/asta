""" asta.constants """
import datetime
from typing import Dict, List, Any
import numpy as np
from asta.dims import Placeholder

_TORCH_IMPORTED = False
try:
    import torch

    _TORCH_IMPORTED = True
except ImportError:
    pass
_TENSORFLOW_IMPORTED = False
try:
    import tensorflow

    _TENSORFLOW_IMPORTED = True
except ImportError:
    pass


# pylint: disable=invalid-name, too-few-public-methods


# Metaclasses.
class NonInstanceMeta(type):
    """ Metaclass for ``NonInstanceType``. """

    def __instancecheck__(cls, inst: Any) -> bool:
        """ No object is an instance of this type. """
        return False


class NonInstanceType(metaclass=NonInstanceMeta):
    """ No object is an instance of this class. """


class TorchModule:
    """ A dummy torch module for when torch is not installed. """

    def __init__(self) -> None:
        self.Tensor = object
        self.Size = NonInstanceType
        self.dtype = NonInstanceType
        self.int32 = NonInstanceType
        self.float32 = NonInstanceType
        self.bool = NonInstanceType
        self.uint8 = NonInstanceType


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


if not _TORCH_IMPORTED:
    torch = TorchModule()


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
    Placeholder,
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
