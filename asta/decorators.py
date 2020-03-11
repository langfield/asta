"""
PRIVATE MODULE: do not import (from) it directly.

This module contains decorators.
"""
import os
import inspect
from typing import Any, Tuple, Dict

import numpy as np

import asta.dims
from asta.dims import Placeholder
from asta.array import Array
from asta._array import _ArrayMeta
from asta.classes import SubscriptableMeta
from asta.constants import torch, Color, _TORCH_IMPORTED


METAMAP: Dict[type, Any] = {_ArrayMeta: Array}
if _TORCH_IMPORTED:
    from asta.tensor import Tensor
    from asta._tensor import _TensorMeta

    METAMAP[_TensorMeta] = Tensor


def refresh(annotation: SubscriptableMeta) -> SubscriptableMeta:
    """ Load an asta type annotation containing placeholders. """
    dtype = annotation.dtype
    shape = annotation.shape
    dimvars = []

    if annotation.shape is not None:
        for dim in annotation.shape:
            if isinstance(dim, Placeholder):
                placeholder = dim
                dimvar = getattr(asta.dims, placeholder.name)

                # Handle case where placeholder is unpacked in annotation.
                if placeholder.unpacked:
                    for elem in dimvar:
                        dimvars.append(elem)
                else:
                    dimvars.append(dimvar)
            else:
                dimvars.append(dim)
        assert len(dimvars) == len(shape)
        shape = tuple(dimvars)

    # Note we're guaranteed that ``annotation`` has type ``SubscriptableMeta``.
    subscriptable_class = METAMAP[type(annotation)]

    if shape is not None:
        annotation = subscriptable_class[dtype, shape]
    else:
        annotation = subscriptable_class[dtype]

    return annotation


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


def typechecked(decorated):  # type: ignore[no-untyped-def]
    """
    Typecheck a function annotated with ``asta`` type objects. This decorator
    will only check the shape and datatype of parameters annotated with
    ``asta`` type variables.  Mypy should be used for everything else. Note
    that the argument-parameter assignment problem is trivial because the only
    arguments which can be out of order are keyword arguments.

    Parameters
    ----------
    decorated : ``Callable[[Any], Any]``.
        The function to be typechecked.

    Returns
    -------
    _wrapper : ``Callable[[Any], Any]``.
        The decorated version of ``decorated``.
    """

    def _wrapper(*args: Tuple[Any], **kwargs: Dict[str, Any]) -> Any:
        """ Decorated/typechecked function. """
        annotations = decorated.__annotations__

        fn_rep = f"{Color.BOLD}{decorated.__module__}.{decorated.__name__}(){Color.END}"
        header = f"<asta::@typechecked::{fn_rep}>"
        min_pad_size = 10
        pad_size = 80 - len(header)
        side_size = max(pad_size // 2, min_pad_size)
        pad_parity = pad_size % 2 if side_size > 10 else 0
        left_padding = "=" * side_size
        right_padding = "=" * (side_size + pad_parity)
        print(f"{left_padding}{header}{right_padding}")

        # Get number of non-return annotations.
        num_annots = len(annotations)
        num_non_return_annots = num_annots
        if "return" in annotations:
            num_non_return_annots -= 1

        sig = inspect.signature(decorated)
        paramlist = list(sig.parameters)

        # Remove unannotated instance/class/metaclass reference.
        checkable_args = args
        refs = ("self", "cls", "mcs")
        if len(sig.parameters) == num_non_return_annots + 1 and paramlist[0] in refs:
            checkable_args = checkable_args[1:]

        num_args = len(checkable_args) + len(kwargs)
        if num_non_return_annots != num_args:
            num_annot_err = f"Mismatch between number of annotated "
            num_annot_err += f"non-(self / cls / mcs) parameters "
            num_annot_err += f"'({num_non_return_annots})' and number of arguments "
            num_annot_err += f"'({num_args})'. There may be a type annotation missing."
            raise TypeError(num_annot_err)

        # Check positional arguments.
        for i, arg in enumerate(checkable_args):

            name = list(annotations.keys())[i]
            annotation = annotations[name]
            if isinstance(annotation, SubscriptableMeta):
                annotation = refresh(annotation)
                rep = type_representation(arg)
                if not isinstance(arg, annotation):
                    type_err = f"Argument value for parameter '{name}' "
                    type_err += f"has wrong type. Expected type: '{annotation}' "
                    type_err += f"Actual type: '{rep}'"
                    raise TypeError(type_err)
                print(
                    f"{Color.GREEN}PASSED{Color.END}: Arg '{name}' matched parameter '{annotation}' with actual type: '{rep}'"
                )

        # Check keyword arguments.
        for name, kwarg in kwargs.items():
            annotation = annotations[name]
            if isinstance(annotation, SubscriptableMeta):
                annotation = refresh(annotation)
                rep = type_representation(kwarg)
                if not isinstance(kwarg, annotation):
                    type_err = f"Argument value for parameter '{name}' "
                    type_err += f"has wrong type. Expected type: '{annotation}' "
                    type_err += f"Actual type: '{rep}'"
                    raise TypeError(type_err)
                print(
                    f"{Color.GREEN}PASSED{Color.END}: Arg '{name}' matched parameter '{annotation}' with actual type: '{rep}'"
                )

        # Check return.
        ret = decorated(*args, **kwargs)
        return_annotation = annotations["return"]
        if isinstance(return_annotation, SubscriptableMeta):
            return_annotation = refresh(return_annotation)
            rep = type_representation(ret)
            if not isinstance(ret, return_annotation):
                type_err = f"Return value has wrong type. "
                type_err += f"Expected type: '{return_annotation}' "
                type_err += f"Actual type: '{rep}'"
                raise TypeError(type_err)
            print(
                f"{Color.GREEN}PASSED{Color.END}: Return matched return type '{return_annotation}' with actual type: '{rep}'"
            )

        return ret

    _wrapper.__module__ = decorated.__module__
    _wrapper.__name__ = decorated.__name__

    if "ASTA_TYPECHECK" in os.environ and os.environ["ASTA_TYPECHECK"] == "1":
        return _wrapper
    return decorated
