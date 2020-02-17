""" Scalar type class for use in annotating scalar ``Array`` types. """
import datetime
from typing import Any, List

import torch
import numpy as np

# pylint: disable=too-few-public-methods


class _ScalarMeta(type):
    """ A meta class for the ``Scalar`` class. """

    _GENERIC_TYPES: List[type]
    _ARRAY_TYPES: List[type]

    def __instancecheck__(cls, inst: Any) -> bool:
        """ Support expected behavior for ``isinstance(<number-like>, Scalar)``. """
        for arr_type in cls._ARRAY_TYPES:
            if isinstance(inst, arr_type):
                assert hasattr(inst, "shape")
                if inst.shape == tuple():  # type: ignore[attr-defined]
                    return True
                return False
        for generic_type in cls._GENERIC_TYPES:
            if isinstance(inst, generic_type):
                return True
        return False


class Scalar(metaclass=_ScalarMeta):
    """ A generic scalar type class. """

    _GENERIC_TYPES: List[type] = [
        bool,
        int,
        float,
        complex,
        bytes,
        str,
        datetime.datetime,
        datetime.timedelta,
    ]
    _ARRAY_TYPES: List[type] = [np.ndarray, torch.Tensor]
