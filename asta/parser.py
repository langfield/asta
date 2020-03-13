#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This module contains a general subscript parser for subscriptable types. """
from typing import Optional, Tuple, Union

from sympy.core.expr import Expr

from asta.utils import is_subtuple
from asta.scalar import Scalar
from asta.classes import SubscriptableMeta
from asta.constants import EllipsisType

# pylint: disable=unidiomatic-typecheck


def parse_subscript(
    cls: SubscriptableMeta,
    item: Union[type, Optional[Union[int, EllipsisType]]],  # type: ignore
    dtype_metaclass: type,
) -> Tuple[Optional[type], Optional[Tuple], str]:
    """ Set class attributes based on the passed dtype/dim data. """
    kind: str = ""
    dtype: Optional[type] = None
    shape: Optional[Tuple] = None

    err = f"Invalid dimension '{item}' of type '{type(item)}'. "
    err += f"Valid dimension types: {cls.DIM_TYPES}"

    if isinstance(item, (type, dtype_metaclass)) and item != Scalar:
        dtype, kind = cls.get_dtype(item)
        shape = None

    # Case where dtype is Any and shape is scalar.
    elif item in (Scalar, ()):
        shape = ()

    # Case where dtype is not passed in, and there's one input.
    # i.e. ``Array[1]`` or ``Array[...]``.
    elif not isinstance(item, tuple):
        if type(item) not in cls.DIM_TYPES:
            raise TypeError(err)
        shape = (item,)

    # Case where ``item`` is a nonempty tuple.
    elif item:

        # Case where generic type is specified.
        if isinstance(item[0], (type, dtype_metaclass)) and item[0] != Scalar:
            dtype, kind = cls.get_dtype(item[0])
            for dim in item[1:]:

                # Treat sympy expressions specially.
                if isinstance(dim, Expr):
                    pass
                elif type(dim) not in cls.DIM_TYPES:
                    err = f"Invalid dimension '{dim}' of type '{type(dim)}'. "
                    err += f"Valid dimension types: {cls.DIM_TYPES}"
                    raise TypeError(err)
            shape = SubscriptableMeta.get_shape(item[1:])

        # Case where generic type is unspecified.
        else:
            for dim in item:
                if type(dim) not in cls.DIM_TYPES:
                    err = f"Invalid dimension '{dim}' of type '{type(dim)}'. "
                    err += f"Valid dimension types: {cls.DIM_TYPES}"
                    raise TypeError(err)
            shape = cls.get_shape(item)
    else:
        empty_err = "Argument to '{cls.NAME}[]' cannot be empty tuple. "
        empty_err += "Use '{cls.NAME}[None]' to indicate a scalar."
        raise TypeError(empty_err)

    if isinstance(shape, tuple) and is_subtuple((..., ...), shape, {})[0]:
        raise TypeError("Invalid shape: repeated '...'")

    return dtype, shape, kind
