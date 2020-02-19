"""
PRIVATE MODULE: do not import (from) it directly.

This module contains decorators.
"""
import os
import inspect
from typing import Callable, Any, Tuple, Dict

from asta._array import _ArrayMeta
from asta._tensor import _TensorMeta


def typechecked(decorated): # type: ignore[no-untyped-def]
    """
    Typecheck a function annotated with ``asta`` type objects. This decorator will only
    check the shape and datatype of parameters annotated with ``asta`` type variables.
    Mypy should be used for everything else.

    Parameters
    ----------
    decorated : ``Callable[[Any], Any]``.
        The function to be typechecked.

    Returns
    -------
    _wrapper : ``Callable[[Any], Any]``.
        The decorated version of ``decorated``.
    """
    sig = inspect.signature(decorated)
    param_names = list(sig.parameters)

    def _wrapper(*args: Tuple[Any], **kwargs: Dict[str, Any]) -> Any:
        """ Decorated/typechecked function. """

        for i, arg in enumerate(args):
            param = sig.parameters[param_names[i]]
            if isinstance(param.annotation, (_ArrayMeta, _TensorMeta)):
                if not isinstance(arg, param.annotation):
                    type_err = f"Argument value '{arg}' for parameter '{param.name}' "
                    type_err += f"has wrong type. Expected type: '{param.annotation}' "
                    type_err += f"Actual type: '{type(arg)}'"
                    raise TypeError(type_err)

        ret = decorated(*args, **kwargs)
        if isinstance(sig.return_annotation, (_ArrayMeta, _TensorMeta)):
            if not isinstance(ret, sig.return_annotation):
                type_err = f"Return value '{ret}' has wrong type. "
                type_err += f"Expected type: '{sig.return_annotation}' "
                type_err += f"Actual type: '{type(ret)}'"
                raise TypeError(type_err)

        return ret

    if os.environ["ASTA_TYPECHECK"] == "1":
        return _wrapper
    return decorated
