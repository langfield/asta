"""
PRIVATE MODULE: do not import (from) it directly.

This module contains decorators.
"""
import os
import inspect
from typing import Any, Tuple, Dict, Set

from sympy import solvers
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol

import asta.dims
from asta.dims import Placeholder
from asta.utils import shapecheck
from asta.array import Array
from asta._array import _ArrayMeta
from asta.classes import SubscriptableMeta
from asta.constants import torch, _TORCH_IMPORTED
from asta.display import (
    type_representation,
    pass_argument,
    pass_return,
    fail_argument,
    fail_return,
    fail_uninitialized,
    get_header,
)


METAMAP: Dict[type, SubscriptableMeta] = {_ArrayMeta: Array}

if _TORCH_IMPORTED:
    from asta.tensor import Tensor
    from asta._tensor import _TensorMeta

    METAMAP[_TensorMeta] = Tensor


def refresh(
    annotation: SubscriptableMeta, halt: bool
) -> Tuple[SubscriptableMeta, bool]:
    """ Load an asta type annotation containing classical placeholders. """
    dtype = annotation.dtype
    shape = annotation.shape
    dimvars = []
    initialized = True
    uninitialized_placeholder_names: Set[str] = set()

    if annotation.shape is not None:
        for item in annotation.shape:

            # Handle fixed placeholders.
            if isinstance(item, Placeholder):
                placeholder = item
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
            else:
                dimvars.append(item)
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


def get_equations(
    equations: Set[Expr],
    annotation: SubscriptableMeta,
    unrefreshed: SubscriptableMeta,
    arg: Any,
) -> Dict[Expr, int]:
    """ TODO: Update: Returns equations with actual values inserted. """

    if annotation.shape is not None:
        assert unrefreshed.shape is not None

        # Handle case where type(shape) != tuple, e.g. ``torch.Size``.
        arg_shape = tuple(arg.shape)
        assert not isinstance(arg_shape, torch.Size)

        # Grab the pieces of the instance shape corresponding to annotation
        # shape elements.
        match, shape_pieces = shapecheck(arg_shape, annotation.shape)
        assert match
        assert len(annotation.shape) == len(unrefreshed.shape) == len(shape_pieces)

        # Iterate over class shape and corresponding instance shape pieces.
        for item, piece in zip(unrefreshed.shape, shape_pieces):

            # If a class shape element is a sympy expression.
            # TODO: Consider changing these checks to use ``core.Basic``.
            if isinstance(item, (Symbol, Expr)):
                vdim = item
                assert len(piece) == 1
                literal: int = piece[0]

                # Create an equation (set equal to zero).
                equation: Expr = item - literal
                equations.add(equation)

    return equations


def check_annotation(
    val: Any, name: str, annotations: Dict[str, Any], equations: Set[Expr]
) -> Dict[Expr, int]:
    """ Check if ``val`` is of type ``annotation`` for asta types only. """
    annotation = annotations[name]
    halt = os.environ["ASTA_TYPECHECK"] == "2"

    # TODO: The solution to the set of equations for each individual annotation
    # should be unique. If there is no solution, print/raise an error. If there
    # are multiple solutions, raise an error; someone is trying to do inference
    # on way too many variables.

    # Thus there should be no need to iterate over the function signature more
    # than once. A single isinstance check should always be sufficient to
    # determine if there is a solution. However, that does not mean that single
    # isinstance checks are self-contained. In addition to specifying a unique
    # solution, all the annotations' solutions must agree for a given function
    # signature.

    # Determine which display functions to use.
    if name == "return":
        fail_fn = fail_return
        pass_fn = pass_return
    else:
        fail_fn = fail_argument
        pass_fn = pass_argument

    # Only check if the annotation is an asta subscriptable class.
    if isinstance(annotation, SubscriptableMeta):
        unrefreshed = annotation
        annotation, initialized = refresh(annotation, halt)
        if not initialized:
            return equations

        # Check if the literal ``val`` matches the annotation.
        rep: str = type_representation(val)
        # HARDCODE
        identifier = "0" if name == "return" else name

        # If the isinstance check fails, print/raise an error.
        # TODO: All the equation logic should be dumped into the isinstance methods.
        if not isinstance(val, annotation):
            fail_fn(identifier, annotation, rep, halt)

        # Otherwise, print a pass.
        else:
            pass_fn(identifier, annotation, rep)

            # Update variable dimension map.
            equations = get_equations(equations, annotation, unrefreshed, val)

    return equations


def validate_annotations(  # type: ignore[no-untyped-def]
    decorated, annotations: Dict[str, Any], args: Tuple[Any], kwargs: Dict[str, Any],
) -> Tuple[Tuple[Any], Dict[str, Any]]:
    """
    Make sure there is an annotation for each parameter, return arguments in
    args other than class references (self, cls, mcs), and keyword arguments in
    kwargs with defaults filled in.
    """
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
    defaulted_kwargs: Dict[str, Any] = defaults.copy()
    defaulted_kwargs.update(kwargs)

    # Remove unannotated instance/class/metaclass reference.
    checkable_args = args
    refs = ("self", "cls", "mcs")
    if len(sig.parameters) == num_non_return_annots + 1 and paramlist[0] in refs:
        checkable_args = checkable_args[1:]  # type: ignore[assignment]

    # Check for mismatch between lengths of arguments/annotations.
    num_args = len(checkable_args) + len(defaulted_kwargs)
    if num_non_return_annots != num_args:
        num_annot_err = f"Mismatch between number of annotated "
        num_annot_err += f"non-(self / cls / mcs) parameters "
        num_annot_err += f"'({num_non_return_annots})' and number of arguments "
        num_annot_err += f"'({num_args})'. There may be a type annotation missing."
        raise TypeError(num_annot_err)

    return checkable_args, defaulted_kwargs


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

        # Print header for ``decorated``.
        header: str = get_header(decorated)
        print(header)

        equations: Set[Expr] = set()
        annotations: Dict[str, Any] = decorated.__annotations__
        checkable_args, defaulted_kwargs = validate_annotations(
            decorated,
            annotations,
            args,  # type: ignore
            kwargs,
        )

        # Consider making one monolithic argument map of names -> args,
        # incorporating args, kwargs, and return, and iterating over only that.

        # Check positional arguments.
        for i, arg in enumerate(checkable_args):
            name = list(annotations.keys())[i]
            equations = check_annotation(arg, name, annotations, equations)

        # Check keyword arguments.
        for name, kwarg in defaulted_kwargs.items():
            equations = check_annotation(kwarg, name, annotations, equations)

        # TODO: Treat tuples, lists, sequences recursively.
        # Check return.
        ret = decorated(*args, **kwargs)
        equations = check_annotation(ret, "return", annotations, equations)

        return ret

    _wrapper.__module__ = decorated.__module__
    _wrapper.__name__ = decorated.__name__

    return _wrapper
