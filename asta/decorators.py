"""
PRIVATE MODULE: do not import (from) it directly.

This module contains decorators.
"""
import os
from typing import Any, Tuple, Dict

import asta.dims
from asta.dims import Placeholder
from asta.array import Array
from asta.tensor import Tensor
from asta._array import _ArrayMeta
from asta.classes import SubscriptableMeta
from asta.constants import _TORCH_IMPORTED


METAMAP: Dict[type, Any] = {_ArrayMeta: Array}
if _TORCH_IMPORTED:
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
                dimvar = getattr(asta.dims, dim.name)
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

        # Check positional arguments.
        for i, arg in enumerate(args):
            name = list(annotations.keys())[i]
            annotation = annotations[name]
            if isinstance(annotation, SubscriptableMeta):
                annotation = refresh(annotation)
                if not isinstance(arg, annotation):
                    type_err = f"Argument value for parameter '{name}' "
                    type_err += f"has wrong type. Expected type: '{annotation}' "
                    type_err += f"Actual type: '{type(arg)}'"
                    raise TypeError(type_err)
                print(f"PASSED: Arg '{name}' matched parameter '{annotation}'.")

        # Check keyword arguments.
        for name, kwarg in kwargs.items():
            annotation = annotations[name]
            if isinstance(annotation, SubscriptableMeta):
                annotation = refresh(annotation)
                if not isinstance(kwarg, annotation):
                    type_err = f"Argument value for parameter '{name}' "
                    type_err += f"has wrong type. Expected type: '{annotation}' "
                    type_err += f"Actual type: '{type(kwarg)}'"
                    raise TypeError(type_err)
                print(f"PASSED: Arg '{name}' matched parameter '{annotation}'.")

        # Check return.
        ret = decorated(*args, **kwargs)
        return_annotation = annotations["return"]
        if isinstance(return_annotation, SubscriptableMeta):
            return_annotation = refresh(return_annotation)
            if not isinstance(ret, return_annotation):
                type_err = f"Return value has wrong type. "
                type_err += f"Expected type: '{return_annotation}' "
                type_err += f"Actual type: '{type(ret)}'"
                raise TypeError(type_err)
            print(f"PASSED: Return matched return type '{return_annotation}'.")

        return ret

    _wrapper.__module__ = decorated.__module__
    _wrapper.__name__ = decorated.__name__

    if "ASTA_TYPECHECK" in os.environ and os.environ["ASTA_TYPECHECK"] == "1":
        return _wrapper
    return decorated
