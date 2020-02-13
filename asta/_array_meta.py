#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This module contains meta functionality for the ``Array`` type. """
import datetime
from typing import List, Optional, Any, Tuple, Dict, Union
from functools import lru_cache

import numpy as np
from typish import SubscriptableType
from typish._types import Ellipsis_, NoneType

# pylint: disable=unidiomatic-typecheck, too-few-public-methods


class _ArrayMeta(SubscriptableType):
    """ A Meta class for the Array class. """

    @lru_cache()
    def __getitem__(cls, item: Any) -> type:
        """ Defer to ``typish``, which calls ``cls._after_subscription()``. """
        return SubscriptableType.__getitem__(cls, item)  # type: ignore

    def __instancecheck__(cls, inst: Any) -> bool:
        """ Support expected behavior for ``isinstance(<array>, Array[<args>])``. """
        assert hasattr(cls, "shape")
        assert hasattr(cls, "dtype")
        assert hasattr(cls, "kind")
        result = False
        if isinstance(inst, np.ndarray):
            result = True  # In case of an empty array or no ``cls.kind``.
            if inst.dtype.names:
                result = False

            print("Kind:", cls.kind)
            print("Dtype:", cls.dtype)
            print("Kind of instance:", inst.dtype.kind)
            print("Dtype of instance:", inst.dtype)

            # If we have ``cls.dtype``, we can be maximally precise.
            if cls.dtype and cls.dtype != inst.dtype:
                result = False

            # Otherwise, check that the kinds match.
            elif cls.kind and cls.kind != inst.dtype.kind:
                result = False

            # If we have a shape and it doesn't match, return False.
            elif cls.shape and cls.shape != inst.shape:
                result = False

        return result


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
    _NP_KINDS = [np.dtype(scalar_type).kind for scalar_type in _GENERIC_TYPES]
    kind: str = ""
    dtype: Optional[np.dtype] = None
    shape: Optional[Tuple[Optional[Union[int, Ellipsis_]]]] = None

    def __new__(cls, *args: Tuple[Any], **kwargs: Dict[str, Any]) -> Any:
        raise TypeError("Cannot instantiate abstract class 'Array'.")

    @staticmethod
    def get_dtype(item: Any) -> Tuple[Optional[np.dtype], Optional[str]]:
        """ Computes dtype and kind if item is a dtype, and kind if it's a type. """
        dtype = None
        kind = None

        # Case where ``item`` is a dtype (``Array[np.dtype("complex128")]``).
        if isinstance(item, np.dtype):
            kind = item.kind
            dtype = item

        # Case where ``item`` is a python3 type (``Array[int]``).
        elif isinstance(item, type):
            generic_type = item
            if item == datetime.datetime:
                generic_type = np.datetime64
            if item == datetime.timedelta:
                generic_type = np.timedelta64
            kind = np.dtype(generic_type).kind

        return dtype, kind

    @classmethod
    def _after_subscription(
        cls, item: Union[type, Optional[Union[int, Ellipsis_]]]
    ) -> None:
        """ Set class attributes based on the passed dtype/dim data. """
        err = f"Invalid dimension '{item}' of type '{type(item)}'. "
        err += f"Valid dimension types: {cls._DIM_TYPES}"

        cls.dtype, cls.kind = _Array.get_dtype(item)
        if isinstance(item, (np.dtype, type)):
            cls.shape = None

        # Case where dtype is not passed in, and there's one input.
        # i.e. ``Array[1]`` or ``Array[None]`` or ``Array[...]``.
        elif not isinstance(item, tuple):
            if type(item) not in cls._DIM_TYPES:
                raise TypeError(err)
            cls.shape = (item,)
        else:

            # Case where generic type is specified.
            if item and isinstance(item[0], (type, np.dtype)):
                cls.dtype, cls.kind = _Array.get_dtype(item[0])
                for i, dim in enumerate(item[1:]):
                    if type(dim) not in cls._DIM_TYPES:
                        raise TypeError(err)
                cls.shape = item[1:]

            # Case where generic type is unspecified.
            else:
                for i, dim in enumerate(item):
                    if type(dim) not in cls._DIM_TYPES:
                        raise TypeError(err)
                cls.shape = item
