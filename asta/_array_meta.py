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
        return SubscriptableType.__getitem__(cls, item) # type: ignore

    def __instancecheck__(cls, inst: Any) -> bool:
        """ Support expected behavior for ``isinstance(<array>, Array[<args>])``. """
        result = False
        if isinstance(inst, np.ndarray):
            result = True  # In case of an empty array or no ``cls._generic_type``.
            rows = 0
            cols = 0
            if len(inst.shape) > 0:
                rows = inst.shape[0]
            if len(inst.shape) > 1:
                cols = inst.shape[1]

            if inst.size > 0 and cls.generic_type:
                if isinstance(cls.generic_type, tuple):
                    inst_dtypes = [inst.dtype[name] for name in inst.dtype.names]
                    cls_dtypes = [np.dtype(typ) for typ in cls.generic_type]
                    result = inst_dtypes == cls_dtypes
                else:
                    result = isinstance(inst[0], cls.generic_type)
                    result |= inst.dtype == np.dtype(cls.generic_type)
                result &= cls.rows is ... or cls.rows == rows
                result &= cls.cols is ... or cls.cols == cols
        return result


class _Array(metaclass=_ArrayMeta):
    """ This class exists to keep the Array class as clean as possible. """

    _DIM_TYPES: List[type] = [int, Ellipsis_, NoneType]
    # TODO: Set default programmatically based on type (array v. tensor).
    generic_type: type = float
    shape: Tuple[Optional[Union[int, Ellipsis_]]]

    def __new__(cls, *args: Tuple[Any], **kwargs: Dict[str, Any]) -> _Array:
        raise TypeError("Cannot instantiate abstract class 'Array'.")

    @classmethod
    def _after_subscription(cls, item: Any) -> None:
        """ Set class attributes based on the passed dtype/dim data. """

        # Case where only the dtype of the array is passed (``Array[int]``).
        if not isinstance(item, tuple):
            cls.generic_type = item
            print(f"Setting generic type attribute to '{item}'.")
        else:

            # Don't allow empty tuples.
            if not item:
                raise TypeError("Parameter Array[...] cannot be empty.")

            # So now ``item`` is a tuple, and it has at least 1 element.
            if isinstance(item[0], type):
                cls.generic_type = item[0]

            # Handle any shape information.
            if len(item) > 1:
                for i, dim in enumerate(item[1:]):
                    if type(dim) not in cls._DIM_TYPES:
                        err = f"Unexpected type {type(dim)}."
                        err += f"Valid dimension types: {cls._DIM_TYPES}"
                        raise TypeError(err)
