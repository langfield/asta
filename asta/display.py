""" Functions for generating typechecker output. """
from typing import Any, Dict, Set, List

import numpy as np
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol

from oxentiel import Oxentiel

from asta.array import Array
from asta.classes import SubscriptableMeta
from asta.constants import torch, tf, Color, _TORCH_IMPORTED, _TENSORFLOW_IMPORTED


if _TORCH_IMPORTED:
    from asta.tensor import Tensor

if _TENSORFLOW_IMPORTED:
    from asta.tftensor import TFTensor

FAIL = f"{Color.RED}FAILED{Color.END}"
PASS = f"{Color.GREEN}PASSED{Color.END}"


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
    elif isinstance(arg, tf.Tensor):
        astatype = TFTensor[dtype, shape]  # type: ignore
        rep = repr(astatype).replace("asta", "tf")
    return rep


def pass_argument(name: str, ann: SubscriptableMeta, rep: str, ox: Oxentiel) -> None:
    """ Print typecheck pass notification for arguments. """
    msg = f"{PASS}: Argument '{name}' matched parameter '{ann}' "
    msg += f"with actual type: '{rep}'"
    handle_pass(msg, ox)


def pass_return(name: str, ann: SubscriptableMeta, rep: str, ox: Oxentiel) -> None:
    """ Print typecheck pass notification for return values. """
    msg = f"{PASS}: Return {name} matched return type '{ann}' with actual type: '{rep}'"
    handle_pass(msg, ox)


def handle_error(err: str, ox: Oxentiel) -> None:
    """ Either print or raise ``err``. """
    if ox.raise_errors:
        raise TypeError(err)
    print(err)


def handle_pass(msg: str, ox: Oxentiel) -> None:
    """ Either print or raise ``err``. """
    if ox.print_passes:
        print(msg)


def fail_argument(name: str, ann: SubscriptableMeta, rep: str, ox: Oxentiel) -> None:
    """ Print/raise typecheck fail error for arguments. """
    err = f"{FAIL}: Argument '{name}' has wrong type. Expected type: '{ann}' "
    err += f"Actual type: '{rep}'"
    handle_error(err, ox)


def fail_return(name: str, ann: SubscriptableMeta, rep: str, ox: Oxentiel) -> None:
    """ Print/raise typecheck fail error for return values. """
    err = f"{FAIL}: Return {name} has wrong type. Expected type: "
    err += f"'{ann}' Actual type: '{rep}'"
    handle_error(err, ox)


def fail_uninitialized(name: str, ox: Oxentiel) -> None:
    """ Print/raise typecheck fail error for uninitialized placeholder. """
    err = f"{FAIL}: Uninitialized placeholder '{name}'"
    handle_error(err, ox)


def fail_tuple(name: str, rep: str, ox: Oxentiel) -> None:
    """ Print/raise error when argument is not a tuple. """
    err = f"{FAIL}: Argument '{name}' must be a Tuple. Actual type: '{rep}'"
    handle_error(err, ox)


def fail_empty_tuple(name: str, rep: str, ox: Oxentiel) -> None:
    """ Print/raise error when argument is not an empty tuple. """
    err = f"{FAIL}: Argument '{name}' must be an empty tuple. Actual type: '{rep}'"
    handle_error(err, ox)


def fail_tuple_length(
    name: str, length: int, ann_rep: str, ann_length: int, ox: Oxentiel
) -> None:
    """ Print/raise error when argument tuple has mismatched length. """
    err = f"{FAIL}: Tuple argument '{name}' has mismatched length ({length}). "
    err += f"Expected type: '{ann_rep}' with length {ann_length}"
    handle_error(err, ox)


def fail_namedtuple(name: str, ann_rep: str, rep: str, ox: Oxentiel) -> None:
    """ Print/raise error when NamedTuple fails isinstance check. """
    err = f"{FAIL}: Argument '{name}' has wrong type. Expected type: '{ann_rep}' "
    err += f"Actual type: '{rep}'"
    handle_error(err, ox)


def fail_list(name: str, ann: Any, rep: str, ox: Oxentiel) -> None:
    """ Print/raise error when List fails isinstance check. """
    err = f"{FAIL}: Argument '{name}' must be a list. Expected type: '{ann}' "
    err += f"Actual type: '{rep}'"
    handle_error(err, ox)


def fail_sequence(name: str, ann: Any, rep: str, ox: Oxentiel) -> None:
    """ Print/raise error when ``collections.abc.Sequence`` fails isinstance check. """
    err = f"{FAIL}: Argument '{name}' must be a sequence. Expected type: '{ann}' "
    err += f"Actual type: '{rep}'"
    handle_error(err, ox)


def fail_dict(name: str, ann: Any, rep: str, ox: Oxentiel) -> None:
    """ Print/raise error when ``Dict[*]`` fails isinstance check. """
    err = f"{FAIL}: Argument '{name}' must be a dictionary. Expected type: '{ann}' "
    err += f"Actual type: '{rep}'"
    handle_error(err, ox)


def fail_set(name: str, ann: Any, rep: str, ox: Oxentiel) -> None:
    """ Print/raise error when ``Set[*]`` fails isinstance check. """
    err = f"{FAIL}: Argument '{name}' must be a set. Expected type: '{ann}' "
    err += f"Actual type: '{rep}'"
    handle_error(err, ox)


def fail_system(
    equations: Set[Expr],
    symbols: Set[Symbol],
    solutions: List[Dict[Symbol, int]],
    ox: Oxentiel,
) -> None:
    """ Print/raise typecheck fail error for uninitialized placeholder. """
    solution_err = f"No unique solution (found {len(solutions)})"
    err = f"{FAIL}: {solution_err} for system "
    err += f"'{equations}' of symbols '{symbols}'"
    handle_error(err, ox)


def fail_fallback(name: str, annrep: str, rep: str, ox: Oxentiel) -> None:
    """ Print/raise error when arbitrary value fails isinstance check. """
    err = f"{FAIL}: Argument '{name}' is not an instance of: '{annrep}' "
    err += f"Actual type: '{rep}'"
    handle_error(err, ox)


def get_header(decorated) -> str:  # type: ignore[no-untyped-def]
    """ Print the typecheck header. """
    core = f"asta::{decorated.__module__}.{decorated.__name__}()"
    bold_core = f"<{Color.BOLD}{core}{Color.END}>"
    min_pad_size = 10
    pad_size = 100 - (len(core) + 2)
    side_size = max(pad_size // 2, min_pad_size)
    pad_parity = pad_size % 2 if side_size > 10 else 0
    left_padding = "=" * side_size
    right_padding = "=" * (side_size + pad_parity)
    header = f"{left_padding}{bold_core}{right_padding}"
    return header
