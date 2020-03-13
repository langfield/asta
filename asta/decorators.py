"""
PRIVATE MODULE: do not import (from) it directly.

This module contains decorators.
"""
import os
import inspect
from typing import Any, Tuple, Dict, Set

import numpy as np

import asta.dims
from asta.dims import Placeholder
from asta.vdims import VariablePlaceholder
from asta.utils import shapecheck
from asta.array import Array
from asta._array import _ArrayMeta
from asta.classes import SubscriptableMeta
from asta.constants import torch, Color, _TORCH_IMPORTED


METAMAP: Dict[type, SubscriptableMeta] = {_ArrayMeta: Array}

if _TORCH_IMPORTED:
    from asta.tensor import Tensor
    from asta._tensor import _TensorMeta

    METAMAP[_TensorMeta] = Tensor


def refresh(
    annotation: SubscriptableMeta, vdims: Dict[VariablePlaceholder, int], halt: bool,
) -> Tuple[SubscriptableMeta, bool]:
    """ Load an asta type annotation containing placeholders. """
    dtype = annotation.dtype
    shape = annotation.shape
    dimvars = []
    initialized = True
    uninitialized_placeholder_names: Set[str] = set()

    if annotation.shape is not None:
        for dim in annotation.shape:
            if isinstance(dim, Placeholder):
                placeholder = dim
                dimvar = getattr(asta.dims, placeholder.name)

                # Catch uninitialized placeholders.
                if isinstance(dimvar, Placeholder):
                    initialized = False
                    name = placeholder.name
                    if name not in uninitialized_placeholder_names:
                        fail_uninitialized(name, annotation, halt=halt)
                    uninitialized_placeholder_names.add(name)

                # Handle case where placeholder is unpacked in annotation.
                if placeholder.unpacked:
                    for elem in dimvar:
                        dimvars.append(elem)
                else:
                    dimvars.append(dimvar)
            elif isinstance(dim, VariablePlaceholder):
                # TODO: Fix naming.
                vdim = dim

                # Add variable dimension if it has already been set.
                if vdim in vdims:
                    dimvar = vdims[vdim]
                    dimvars.append(dimvar)

                # Otherwise, add a wildcard.
                else:
                    dimvars.append(vdim)
            else:
                dimvars.append(dim)
        assert len(dimvars) == len(shape)
        shape = tuple(dimvars)

    # Note we're guaranteed that ``annotation`` has type ``SubscriptableMeta``.
    subscriptable_class = METAMAP[type(annotation)]

    refreshed_annotation: SubscriptableMeta
    if shape is not None:
        refreshed_annotation = subscriptable_class[dtype, shape]  # type: ignore
    else:
        refreshed_annotation = subscriptable_class[dtype]

    return refreshed_annotation, initialized


def update_vplaceholders(
    vdims: Dict[VariablePlaceholder, int],
    annotation: SubscriptableMeta,
    unrefreshed: SubscriptableMeta,
    arg: Any,
) -> Dict[VariablePlaceholder, int]:
    """ Returns an updated copy of vdims with actual values inserted. """
    # Copy the input vdims.
    new_vdims: Dict[VariablePlaceholder, int] = vdims.copy()

    if annotation.shape is not None:
        assert unrefreshed.shape is not None

        # Handle case where type(shape) != tuple.
        arg_shape = tuple(arg.shape)
        assert not isinstance(arg_shape, torch.Size)

        # Grab the pieces of the instance shape corresponding to annotation
        # shape elements.
        match, shape_pieces = shapecheck(arg_shape, annotation.shape)
        assert match
        assert len(annotation.shape) == len(unrefreshed.shape) == len(shape_pieces)

        # Iterate over class shape and corresponding instance shape pieces.
        for dim, piece in zip(unrefreshed.shape, shape_pieces):

            # If a class shape element is a variable placeholder.
            if isinstance(dim, VariablePlaceholder):
                vplaceholder = dim
                assert len(piece) == 1
                literal: int = piece[0]

                # Attempt to update the vdims map.
                if vplaceholder not in new_vdims:
                    new_vdims[vplaceholder] = literal
                elif vplaceholder in new_vdims and new_vdims[vplaceholder] != literal:
                    raise TypeError("This should never happen.")

    return new_vdims


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


def pass_return(i: int, ann: SubscriptableMeta, rep: str) -> None:
    """ Print typecheck pass notification for return values. """
    passed = f"{Color.GREEN}PASSED{Color.END}"
    print(f"{passed}: Return {i} matched return type '{ann}' with actual type: '{rep}'")


def fail_return(i: int, ann: SubscriptableMeta, rep: str, halt: bool) -> None:
    """ Print/raise typecheck fail error for return values. """
    failed = f"{Color.RED}FAILED{Color.END}"
    type_err = f"{failed}: Return {i} has wrong type. Expected type: "
    type_err += f"'{ann}' Actual type: '{rep}'"
    if halt:
        raise TypeError(type_err)
    print(type_err)


def pass_argument(name: str, ann: SubscriptableMeta, rep: str) -> None:
    """ Print typecheck pass notification for arguments. """
    passed = f"{Color.GREEN}PASSED{Color.END}"
    print(f"{passed}: Arg '{name}' matched parameter '{ann}' with actual type: '{rep}'")


def fail_argument(name: str, ann: SubscriptableMeta, rep: str, halt: bool) -> None:
    """ Print/raise typecheck fail error for arguments. """
    failed = f"{Color.RED}FAILED{Color.END}"
    type_err = f"{failed}: Argument value for parameter '{name}' "
    type_err += f"has wrong type. Expected type: '{ann}' "
    type_err += f"Actual type: '{rep}'"
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
    if "ASTA_TYPECHECK" not in os.environ or os.environ["ASTA_TYPECHECK"] == "0":
        return decorated

    def _wrapper(*args: Tuple[Any], **kwargs: Dict[str, Any]) -> Any:
        """ Decorated/typechecked function. """

        # Print the typecheck header.
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
        annotations = decorated.__annotations__
        num_annots = len(annotations)
        num_non_return_annots = num_annots
        if "return" in annotations:
            num_non_return_annots -= 1

        # Get the parameter list from the function signature.
        sig = inspect.signature(decorated)
        paramlist = list(sig.parameters)

        # Get a list of kwargs with defaults filled-in.
        defaults: Dict[str, Any] = {}
        for k, v in sig.parameters.items():
            if v.default is not inspect.Parameter.empty:
                defaults[k] = v.default
        filled_kwargs: Dict[str, Any] = defaults.copy()
        filled_kwargs.update(kwargs)

        # Remove unannotated instance/class/metaclass reference.
        checkable_args = args
        refs = ("self", "cls", "mcs")
        if len(sig.parameters) == num_non_return_annots + 1 and paramlist[0] in refs:
            checkable_args = checkable_args[1:]

        # Check for mismatch between lengths of arguments/annotations.
        num_args = len(checkable_args) + len(filled_kwargs)
        if num_non_return_annots != num_args:
            num_annot_err = f"Mismatch between number of annotated "
            num_annot_err += f"non-(self / cls / mcs) parameters "
            num_annot_err += f"'({num_non_return_annots})' and number of arguments "
            num_annot_err += f"'({num_args})'. There may be a type annotation missing."
            raise TypeError(num_annot_err)

        vdims: Dict[VariablePlaceholder, int] = {}

        halt = os.environ["ASTA_TYPECHECK"] == "2"

        # Check positional arguments.
        for i, arg in enumerate(checkable_args):
            name = list(annotations.keys())[i]
            annotation = annotations[name]

            # Check if the annotation is an asta class.
            if isinstance(annotation, SubscriptableMeta):
                unrefreshed = annotation
                annotation, initialized = refresh(annotation, vdims, halt)
                if not initialized:
                    continue

                # Check if the argument matches the annotation.
                rep = type_representation(arg)
                if not isinstance(arg, annotation):
                    fail_argument(name, annotation, rep, halt)
                else:
                    pass_argument(name, annotation, rep)

                    # Update variable dimension map.
                    vdims = update_vplaceholders(vdims, annotation, unrefreshed, arg)

        # Check keyword arguments.
        for name, kwarg in filled_kwargs.items():
            annotation = annotations[name]
            if isinstance(annotation, SubscriptableMeta):
                unrefreshed = annotation
                annotation, initialized = refresh(annotation, vdims, halt)
                if not initialized:
                    continue
                rep = type_representation(kwarg)
                if not isinstance(kwarg, annotation):
                    fail_argument(name, annotation, rep, halt)
                else:
                    pass_argument(name, annotation, rep)
                    vdims = update_vplaceholders(vdims, annotation, unrefreshed, kwarg)

        # Check return.
        ret = decorated(*args, **kwargs)
        return_annotation = annotations["return"]
        if isinstance(return_annotation, SubscriptableMeta):
            unrefreshed = return_annotation
            return_annotation, initialized = refresh(return_annotation, vdims, halt)
            if initialized:
                rep = type_representation(ret)
                if not isinstance(ret, return_annotation):
                    fail_return(0, return_annotation, rep, halt)
                else:
                    pass_return(0, return_annotation, rep)
                    vdims = update_vplaceholders(
                        vdims, return_annotation, unrefreshed, ret
                    )

        # TODO: Treat tuples, lists, sequences recursively.

        return ret

    _wrapper.__module__ = decorated.__module__
    _wrapper.__name__ = decorated.__name__

    return _wrapper
