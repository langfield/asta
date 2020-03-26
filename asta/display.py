""" Functions for generating typechecker output. """
import inspect
from typing import Any, Dict, Set, List, FrozenSet

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


def get_type_name(type_: type) -> str:
    """ ``typing.*`` types don't have a __name__ on Python 3.7+. """
    # pylint: disable=protected-access
    name: str = getattr(type_, "__name__", None)
    if name is None:
        name = type_._name
    return name


def qualified_name(obj: Any) -> str:
    """
    Return the qualified name (e.g. package.module.Type) for the given object.

    Builtins and types from the :mod:`typing` package get special treatment by having the module
    name stripped from the generated name.

    """
    type_ = obj if inspect.isclass(obj) else type(obj)
    module = type_.__module__
    qualname: str = type_.__qualname__
    return qualname if module in ("typing", "builtins") else f"{module}.{qualname}"


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


def fail_callable(name: str, ann: Any, rep: str, ox: Oxentiel) -> None:
    """ Print/raise error when ``Callable[*]`` argument is not callable. """
    err = f"{FAIL}: Argument '{name}' is not callable. Expected type: '{ann}' "
    err += f"Actual type: '{rep}'"
    handle_error(err, ox)


def fail_too_many_args(
    name: str, expected_num_args: int, declared_num_args: int, ox: Oxentiel
) -> None:
    """ Print/raise error when ``Callable[*]`` argument has too few arguments. """
    err = f"{FAIL}: Callable '{name}' has too many arguments in its declaration. "
    err += f"Exepected: {expected_num_args} Declared: {declared_num_args}"
    handle_error(err, ox)


def fail_too_few_args(
    name: str, expected_num_args: int, declared_num_args: int, ox: Oxentiel
) -> None:
    """ Print/raise error when ``Callable[*]`` argument has too few arguments. """
    err = f"{FAIL}: Callable '{name}' has too few arguments in its declaration. "
    err += f"Exepected: {expected_num_args} Declared: {declared_num_args}"
    handle_error(err, ox)


def fail_protocol(name: str, ann_rep: str, rep: str, ox: Oxentiel) -> None:
    """ Print/raise error when ``Protocol[*]`` fails issubclass check. """
    err = f"{FAIL}: Argument '{name}' of type {rep} is not compatible "
    err += f"with the {ann_rep} protocol."
    handle_error(err, ox)


def fail_union(name: str, typelist: str, rep: str, ox: Oxentiel) -> None:
    """ Print/raise error when ``Union[*]`` fails typecheck. """
    err = f"{FAIL}: Argument '{name}' must be one of 'Union[{typelist}]'. "
    err += f"Actual type: '{rep}'"
    handle_error(err, ox)


def fail_class(name: str, ann: str, rep: str, ox: Oxentiel) -> None:
    """ Print/raise error when argument should be a class but is not of type type. """
    err = f"{FAIL}: Argument '{name}' must be a class. Expected type: '{ann}' "
    err += f"Actual type: '{rep}'"
    handle_error(err, ox)


def fail_subclass(name: str, supercls: str, rep: str, ox: Oxentiel) -> None:
    """ Print/raise error when argument should be subclass of ``expected_class``. """
    err = f"{FAIL}: Argument '{name}' should be a subclass of '{supercls}' "
    err += f"Actual type: '{rep}'"
    handle_error(err, ox)


def fail_complex(name: str, rep: str, ox: Oxentiel) -> None:
    """ Print/raise error when argument should be a complex number. """
    err = f"{FAIL}: Argument '{name}' should be a complex number. "
    err += f"Actual type: '{rep}'"
    handle_error(err, ox)


def fail_float(name: str, rep: str, ox: Oxentiel) -> None:
    """ Print/raise error when argument should be a floating point number. """
    err = f"{FAIL}: Argument '{name}' should be a float. "
    err += f"Actual type: '{rep}'"
    handle_error(err, ox)


def fail_text_io(name: str, rep: str, ox: Oxentiel) -> None:
    """ Print/raise error when argument should be a text-based I/O object. """
    err = f"{FAIL}: Argument '{name}' should be a text-based I/O object. "
    err += f"Actual type: '{rep}'"
    handle_error(err, ox)


def fail_binary_io(name: str, rep: str, ox: Oxentiel) -> None:
    """ Print/raise error when argument should be a binary I/O object. """
    err = f"{FAIL}: Argument '{name}' should be a binary I/O object. "
    err += f"Actual type: '{rep}'"
    handle_error(err, ox)


def fail_io(name: str, rep: str, ox: Oxentiel) -> None:
    """ Print/raise error when argument should be an I/O object. """
    err = f"{FAIL}: Argument '{name}' should be an I/O object. "
    err += f"Actual type: '{rep}'"
    handle_error(err, ox)


def fail_literal(name: str, options: List[Any], arg: Any, ox: Oxentiel) -> None:
    """ Print/raise error when argument should match a given literal. """
    err = f"{FAIL}: Argument '{name}' should be one of '{options}' "
    err += f"Actual value: '{arg}'"
    handle_error(err, ox)


def fail_keys(
    name: str,
    expected_keys: FrozenSet[Any],
    existing_keys: FrozenSet[Any],
    ox: Oxentiel,
) -> None:
    """ Print/raise error when typed dict argument has extra or missing keys. """
    err = f"{FAIL}: Argument '{name}' should be a typed dictionary with keys: "
    err += f"'{expected_keys}'. Actual keys: '{existing_keys}'"
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
    core = f"asta::{decorated.__module__}.{decorated.__qualname__}()"
    bold_core = f"<{Color.BOLD}{core}{Color.END}>"
    min_pad_size = 10
    pad_size = 100 - (len(core) + 2)
    side_size = max(pad_size // 2, min_pad_size)
    pad_parity = pad_size % 2 if side_size > 10 else 0
    left_padding = "=" * side_size
    right_padding = "=" * (side_size + pad_parity)
    header = f"{left_padding}{bold_core}{right_padding}"
    return header
