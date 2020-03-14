""" Functions for generating typechecker output. """
from typing import Any, Dict

import numpy as np

from asta.array import Array
from asta._array import _ArrayMeta
from asta.classes import SubscriptableMeta
from asta.constants import torch, Color, _TORCH_IMPORTED


METAMAP: Dict[type, SubscriptableMeta] = {_ArrayMeta: Array}

if _TORCH_IMPORTED:
    from asta.tensor import Tensor
    from asta._tensor import _TensorMeta

    METAMAP[_TensorMeta] = Tensor


def type_representation(arg: Any) -> str:
    """ Get a string representation of an argument including dtype and shape. """
    rep = repr(type(arg))
    if hasattr(arg, "shape"):
        shape = arg.shape
    if hasattr(arg, "dtype"):
        dtype = arg.dtype
    if isinstance(arg, np.ndarray):
        astatype = Array[dtype, shape]  # type: ignore[valid-type, type-arg]
        rep = repr(astatype).replace("asta", "numpy")
    elif isinstance(arg, torch.Tensor):
        astatype = Tensor[dtype, shape]  # type: ignore
        rep = repr(astatype).replace("asta", "torch")
    return rep


def pass_argument(name: str, ann: SubscriptableMeta, rep: str) -> None:
    """ Print typecheck pass notification for arguments. """
    passed = f"{Color.GREEN}PASSED{Color.END}"
    print(f"{passed}: Arg '{name}' matched parameter '{ann}' with actual type: '{rep}'")


def pass_return(name: str, ann: SubscriptableMeta, rep: str) -> None:
    """ Print typecheck pass notification for return values. """
    passed = f"{Color.GREEN}PASSED{Color.END}"
    print(
        f"{passed}: Return {name} matched return type '{ann}' with actual type: '{rep}'"
    )


def fail_argument(name: str, ann: SubscriptableMeta, rep: str, halt: bool) -> None:
    """ Print/raise typecheck fail error for arguments. """
    failed = f"{Color.RED}FAILED{Color.END}"
    type_err = f"{failed}: Argument value for parameter '{name}' "
    type_err += f"has wrong type. Expected type: '{ann}' "
    type_err += f"Actual type: '{rep}'"
    if halt:
        raise TypeError(type_err)
    print(type_err)


def fail_return(name: str, ann: SubscriptableMeta, rep: str, halt: bool) -> None:
    """ Print/raise typecheck fail error for return values. """
    failed = f"{Color.RED}FAILED{Color.END}"
    type_err = f"{failed}: Return {name} has wrong type. Expected type: "
    type_err += f"'{ann}' Actual type: '{rep}'"
    if halt:
        raise TypeError(type_err)
    print(type_err)


def fail_uninitialized(name: str, ann: SubscriptableMeta, halt: bool) -> None:
    """ Print/raise typecheck fail error for uninitialized placeholder. """
    failed = f"{Color.RED}FAILED{Color.END}"
    type_err = f"{failed}: Uninitialized placeholder '{name}'"
    if halt:
        raise TypeError(type_err)
    print(type_err)


def get_header(decorated) -> str:  # type: ignore[no-untyped-def]
    """ Print the typecheck header. """
    fn_rep = f"{Color.BOLD}{decorated.__module__}.{decorated.__name__}(){Color.END}"
    core = f"<asta::@typechecked::{fn_rep}>"
    min_pad_size = 10
    pad_size = 80 - len(core)
    side_size = max(pad_size // 2, min_pad_size)
    pad_parity = pad_size % 2 if side_size > 10 else 0
    left_padding = "=" * side_size
    right_padding = "=" * (side_size + pad_parity)
    header = f"{left_padding}{core}{right_padding}"
    return header
