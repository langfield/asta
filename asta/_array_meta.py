#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This module contains meta functionality for the ``Array`` type. """
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
        assert hasattr(cls, "generic_type")
        assert hasattr(cls, "shape")
        result = False
        if isinstance(inst, np.ndarray):
            result = True  # In case of an empty array or no ``cls.generic_type``.
            if inst.dtype.names:
                result = False

            print("Generic type:", cls.generic_type)
            print("Generic np dtype:", np.dtype(cls.generic_type))
            print("Dtype of instance:", inst.dtype)
            # elif
            if cls.generic_type and np.dtype(cls.generic_type) != inst.dtype:
                result = False
            if cls.shape:
                result = result and (inst.shape == cls.shape)

        return result


class _Array(metaclass=_ArrayMeta):
    """ This class exists to keep the Array class as clean as possible. """

    _DIM_TYPES: List[type] = [int, Ellipsis_, NoneType]
    generic_type: Optional[Union[type, np.dtype]] = None
    shape: Optional[Tuple[Optional[Union[int, Ellipsis_]]]] = None

    def __new__(cls, *args: Tuple[Any], **kwargs: Dict[str, Any]) -> Any:
        raise TypeError("Cannot instantiate abstract class 'Array'.")

    @classmethod
    def _after_subscription(
        cls, item: Union[type, Optional[Union[int, Ellipsis_]]]
    ) -> None:
        """ Set class attributes based on the passed dtype/dim data. """
        err = f"Invalid dimension '{item}' of type '{type(item)}'. "
        err += f"Valid dimension types: {cls._DIM_TYPES}"

        # Case where only the dtype of the array is passed (``Array[int]``).
        if isinstance(item, (type, np.dtype)):
            cls.generic_type = item

            # If shape is unspecified, set to None.
            cls.shape = None

        # Treat the case where dtype is not passed in, and there's one input.
        # i.e. ``Array[1]`` or ``Array[None]`` or ``Array[...]``.
        elif not isinstance(item, tuple):
            if type(item) not in cls._DIM_TYPES:
                raise TypeError(err)
            cls.shape = (item,)
        else:

            # Case where generic type is specified.
            if item and isinstance(item[0], (type, np.dtype)):
                for i, dim in enumerate(item[1:]):
                    if type(dim) not in cls._DIM_TYPES:
                        raise TypeError(err)
                cls.generic_type = item[0]
                cls.shape = item[1:]

            # Case where generic type is unspecified.
            else:
                for i, dim in enumerate(item):
                    if type(dim) not in cls._DIM_TYPES:
                        raise TypeError(err)
                cls.generic_type = None
                cls.shape = item
