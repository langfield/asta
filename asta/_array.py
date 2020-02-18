#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This module contains meta functionality for the ``Array`` type. """
import datetime
from typing import List, Optional, Any, Tuple, Dict, Union
from functools import lru_cache

import numpy as np

# from typish import SubscriptableType
from asta._classes import SubscriptableMeta, SubscriptableType
from asta.utils import is_subtuple, split, wildcard_eq
from typish._types import Ellipsis_, NoneType


# pylint: disable=unidiomatic-typecheck, too-few-public-methods, too-many-nested-blocks


class _ArrayMeta(SubscriptableMeta):
    """ A meta class for the ``Array`` class. """

    kind: str
    shape: tuple
    dtype: np.dtype

    @lru_cache()
    def __getitem__(cls, item: Any) -> SubscriptableType:
        """ Defer to ``typish``, which calls ``cls._after_subscription()``. """
        return SubscriptableMeta.__getitem__(cls, item)

    def __instancecheck__(cls, inst: Any) -> bool:
        """ Support expected behavior for ``isinstance(<array>, Array[<args>])``. """
        assert hasattr(cls, "kind")
        assert hasattr(cls, "shape")
        assert hasattr(cls, "dtype")
        match = False
        if isinstance(inst, np.ndarray):
            match = True  # In case of an empty array or no ``cls.kind``.
            if inst.dtype.names:
                match = False

            if cls.kind and cls.kind != inst.dtype.kind:
                match = False

            # If we have ``cls.dtype``, we can be maximally precise.
            elif cls.dtype and cls.dtype != inst.dtype and cls.kind == "":
                match = False

            # Handle ellipses.
            elif cls.shape is not None:
                if Ellipsis not in cls.shape and -1 not in cls.shape:
                    if not wildcard_eq(cls.shape, inst.shape):
                        match = False
                elif inst.shape == tuple() != cls.shape:
                    match = False
                else:
                    if is_subtuple((Ellipsis, Ellipsis), cls.shape)[0]:
                        raise TypeError("Invalid shape: repeated '...'")

                    # Determine if/where '...' bookends ``cls.shape``.
                    left_bookend = False
                    right_bookend = False
                    ellipsis_positions: List[int] = []
                    for i, elem in enumerate(cls.shape):
                        if elem == Ellipsis:

                            # e.g. ``Array[..., 1, 2, 3]``.
                            if i == 0:
                                left_bookend = True

                            # e.g. ``Array[1, 2, 3, ...]``.
                            if i == len(cls.shape) - 1:
                                right_bookend = True
                            ellipsis_positions.append(i)

                    # Analogous to ``str.split(<elem>)``, we split the shape on '...'.
                    frags: List[Tuple[int, ...]] = split(cls.shape, Ellipsis)

                    # Cut off end if '...' is there.
                    ishape = inst.shape
                    if left_bookend:
                        ishape = ishape[1:]
                    if right_bookend:
                        ishape = ishape[:-1]

                    for i, frag in enumerate(frags):
                        is_sub, index = is_subtuple(frag, ishape)

                        # Must have ``frag`` contained in ``ishape``.
                        if not is_sub:
                            match = False
                            break

                        # First fragment must start at 0 if '...' is not the first
                        # element of ``cls.shape``.
                        if i == 0 and not left_bookend and index != 0:
                            match = False
                            break

                        # Last fragement must end at (exclusive) ``len(ishape)`` if
                        # '...' is not the last element of ``cls.shape``.
                        if (
                            i == len(frags) - 1
                            and not right_bookend
                            and index + len(frag) != len(ishape)
                        ):
                            match = False
                            break

                        new_start = index + len(frag) + 1
                        ishape = ishape[new_start:]

        return match


class _Array(metaclass=_ArrayMeta):
    """ This class exists to keep the Array class as clean as possible. """

    _DIM_TYPES: List[type] = [int, Ellipsis_, NoneType]
    _GENERIC_TYPES: List[type] = [
        bool,
        int,
        float,
        complex,
        bytes,
        str,
        object,
        np.datetime64,
        np.timedelta64,
    ]
    _UNSIZED_TYPE_KINDS: Dict[type, str] = {bytes: "S", str: "U", object: "O"}
    kind: str = ""
    dtype: Optional[np.dtype] = None
    shape: Optional[Tuple] = None

    def __new__(cls, *args: Tuple[Any], **kwargs: Dict[str, Any]) -> Any:
        raise TypeError("Cannot instantiate abstract class 'Array'.")

    @classmethod
    def get_dtype(cls, item: Any) -> Tuple[Optional[np.dtype], str]:
        """ Computes dtype. """
        dtype = None
        kind = ""

        # Case where ``item`` is a dtype (``Array[np.dtype("complex128")]``).
        if isinstance(item, np.dtype):
            dtype = item

        # Case where ``item`` is a python3 type (``Array[int]``).
        elif isinstance(item, type):
            if item in cls._UNSIZED_TYPE_KINDS:
                generic_type = item
                kind = cls._UNSIZED_TYPE_KINDS[item]
            elif item == datetime.datetime:
                generic_type = np.datetime64
            elif item == datetime.timedelta:
                generic_type = np.timedelta64
            else:
                generic_type = item
            dtype = np.dtype(generic_type)

        return dtype, kind

    @staticmethod
    def get_shape(item: Tuple) -> Optional[Tuple]:
        """ Compute shape from a shape tuple argument. """
        shape: Optional[Tuple] = None
        if item:
            if None not in item:
                shape = item
            elif item == (None,):
                shape = ()
            else:
                none_err = "Too many 'None' arguments. "
                none_err += "Use 'Array[None]' for scalar arrays."
                raise TypeError(none_err)

        return shape

    @classmethod
    def _after_subscription(
        cls, item: Union[type, Optional[Union[int, Ellipsis_]]]
    ) -> None:
        """ Set class attributes based on the passed dtype/dim data. """

        err = f"Invalid dimension '{item}' of type '{type(item)}'. "
        err += f"Valid dimension types: {cls._DIM_TYPES}"

        # Case where dtype is Any and shape is scalar.
        if item is None:
            cls.shape = ()

        elif isinstance(item, (np.dtype, type)):
            cls.dtype, cls.kind = _Array.get_dtype(item)
            cls.shape = None

        # Case where dtype is not passed in, and there's one input.
        # i.e. ``Array[1]`` or ``Array[None]`` or ``Array[...]``.
        elif not isinstance(item, tuple):
            if type(item) not in cls._DIM_TYPES:
                raise TypeError(err)
            cls.shape = (item,)

        # Case where ``item`` is a nonempty tuple.
        elif item:

            # Case where generic type is specified.
            if isinstance(item[0], (type, np.dtype)):
                cls.dtype, cls.kind = _Array.get_dtype(item[0])
                for i, dim in enumerate(item[1:]):
                    if type(dim) not in cls._DIM_TYPES:
                        raise TypeError(err)
                cls.shape = _Array.get_shape(item[1:])

            # Case where generic type is unspecified.
            else:
                for i, dim in enumerate(item):
                    if type(dim) not in cls._DIM_TYPES:
                        raise TypeError(err)
                cls.shape = _Array.get_shape(item)
        else:
            empty_err = "Argument to 'Array[]' cannot be empty tuple. "
            empty_err += "Use 'Array[None]' to indicate a scalar."
            raise TypeError(empty_err)

        if cls.shape is not None and is_subtuple((Ellipsis, Ellipsis), cls.shape)[0]:
            raise TypeError("Invalid shape: repeated '...'")
