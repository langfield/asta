#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This module contains a general subscript parser for subscriptable types. """
from typing import Optional, Tuple, Union, Dict, Any

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol

from asta.utils import is_subtuple
from asta.scalar import Scalar
from asta.classes import SubscriptableMeta
from asta.constants import EllipsisType

# pylint: disable=unidiomatic-typecheck


def verify_dimensions(cls: SubscriptableMeta, item: Any) -> None:
    """ Raises a TypeError if the type of a dimension is invalid. """
    if not isinstance(item, tuple):
        item = (item,)

    for dim in item:

        # Treat sympy expressions specially.
        if isinstance(dim, (Expr, Symbol)):
            pass
        elif type(dim) not in cls.DIM_TYPES:
            err = f"Invalid dimension '{dim}' of type '{type(dim)}'. "
            err += f"Valid dimension types: {cls.DIM_TYPES}"
            raise TypeError(err)


def parse_subscript(
    cls: SubscriptableMeta,
    item: Union[type, Optional[Union[int, EllipsisType]]],  # type: ignore
    dtype_metaclass: type,
) -> Tuple[Optional[type], Optional[Tuple], Optional[Dict[str, Any]], str]:
    """ Set class attributes based on the passed dtype/dim data. """
    kind: str = ""
    if hasattr(cls, "kind"):
        kind = cls.kind
    dtype: Optional[type] = cls.dtype
    shape: Optional[Tuple] = cls.shape
    kwattrs: Optional[Dict[str, Any]] = cls.kwattrs

    isscalar = hash(item) == hash(Scalar)
    if isinstance(item, (type, dtype_metaclass)) and not isscalar:
        dtype, kind = cls.get_dtype(item)
        shape = None

    # Adding support for kwargs.
    elif isinstance(item, dict):
        kwattrs = item
        shape = None

    # Case where dtype is Any and shape is scalar.
    elif item in (Scalar, ()):
        shape = ()

    # Case where dtype is not passed in, and there's one input.
    # i.e. ``Array[1]`` or ``Array[...]``.
    elif not isinstance(item, tuple):

        verify_dimensions(cls, item)
        shape = (item,)

    # Case where ``item`` is a nonempty tuple.
    elif item:

        # Case where generic type is specified.
        isscalar = hash(item[0]) == hash(Scalar)
        if isinstance(item[0], (type, dtype_metaclass)) and not isscalar:
            dtype, kind = cls.get_dtype(item[0])
            item = item[1:]

        # Adding support for keyword attributes.
        if item and isinstance(item[-1], dict):
            kwattrs = item[-1]
            item = item[:-1]

        if item:
            verify_dimensions(cls, item)
            shape = SubscriptableMeta.get_shape(item)

    else:
        # TODO: Update this error message.
        empty_err = "Argument to '{cls.NAME}[]' cannot be empty tuple. "
        empty_err += "Use '{cls.NAME}[None]' to indicate a scalar."
        raise TypeError(empty_err)

    if isinstance(shape, tuple) and is_subtuple((..., ...), shape, set())[0]:
        raise TypeError("Invalid shape: repeated '...'")

    return dtype, shape, kwattrs, kind
