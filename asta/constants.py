""" asta.constants """
import datetime
from typing import Dict, List, Union, Any

import numpy as np
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol

from asta.dims import Placeholder

_TORCH_IMPORTED = False
try:
    import torch

    _TORCH_IMPORTED = True
except ImportError:
    pass
_TENSORFLOW_IMPORTED = False
try:
    import tensorflow as tf

    _TENSORFLOW_IMPORTED = True
except ImportError:
    pass


# pylint: disable=invalid-name, too-few-public-methods


# Classes and metaclasses.
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
        self.Tensor = NonInstanceType
        self.Size = NonInstanceType
        self.dtype = NonInstanceType
        self.int32 = NonInstanceType
        self.float32 = NonInstanceType
        self.bool = NonInstanceType
        self.uint8 = NonInstanceType


class DTypes:
    """ A dummy dtypes attribute object for when tf is not installed. """

    def __init__(self) -> None:
        self.DType = NonInstanceType


class TFModule:
    """ A dummy tensorflow module for when tf is not installed. """

    def __init__(self) -> None:
        self.Tensor = NonInstanceType
        self.TensorShape = NonInstanceType
        self.dtypes = DTypes()
        self.dtypes.DType = NonInstanceType
        self.int64 = NonInstanceType
        self.float64 = NonInstanceType
        self.bool = NonInstanceType
        self.complex128 = NonInstanceType
        self.string = NonInstanceType


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


class Printable:
    """ Class for printing object representations in nested objects. """

    def __init__(self, rep: str):
        self.rep = rep

    def __repr__(self) -> str:
        """ Just return ``rep``. """
        return self.rep


class Color:
    """ Terminal color string literals. """

    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


if not _TORCH_IMPORTED:
    torch = TorchModule()
    tf = TFModule()

# Types.
GenericArray = Union[np.ndarray, torch.Tensor, tf.Tensor]
ARRAY_TYPES: List[type] = [np.ndarray, torch.Tensor, tf.Tensor]
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

CORE_DIM_TYPES: List[type] = [
    int,
    ScalarMeta,
    EllipsisType,
    NoneType,  # type: ignore[misc]
    tuple,
    Placeholder,
    Expr,
    Symbol,
]
NUMPY_DIM_TYPES: List[type] = CORE_DIM_TYPES
TORCH_DIM_TYPES: List[type] = CORE_DIM_TYPES + [torch.Size]
TF_DIM_TYPES: List[type] = CORE_DIM_TYPES + [tf.TensorShape]
NP_UNSIZED_TYPE_KINDS: Dict[type, str] = {bytes: "S", str: "U", object: "O"}
TORCH_DTYPE_MAP: Dict[type, torch.dtype] = {
    int: torch.int32,
    float: torch.float32,
    bool: torch.bool,
    bytes: torch.uint8,
}
TF_DTYPE_MAP: Dict[type, tf.dtypes.DType] = {
    int: tf.int32,
    float: tf.float32,
    complex: tf.complex128,
    bool: tf.bool,
    bytes: tf.string,
    str: tf.string,
}
TF_DTYPES = [
    tf.bfloat16,
    tf.bool,
    tf.complex,
    tf.complex128,
    tf.complex64,
    tf.double,
    tf.float16,
    tf.float32,
    tf.float64,
    tf.half,
    tf.int16,
    tf.int32,
    tf.int64,
    tf.int8,
    tf.qint16,
    tf.qint32,
    tf.qint8,
    tf.quint16,
    tf.quint8,
    tf.resource,
    tf.string,
    tf.uint16,
    tf.uint32,
    tf.uint64,
    tf.uint8,
    tf.variant,
]
