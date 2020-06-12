#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This module contains meta functionality for the ``Tensor`` type. """
from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Union, Optional

import numpy as np
import torch

from asta.utils import attrcheck, shapecheck
from asta.parser import parse_subscript
from asta.classes import GenericMeta, SubscriptableMeta
from asta.constants import TORCH_DIM_TYPES, TORCH_DTYPE_MAP, EllipsisType

# pylint: disable=unidiomatic-typecheck, too-few-public-methods, too-many-nested-blocks


class _TensorMeta(SubscriptableMeta):
    """ A meta class for the ``Tensor`` class. """

    shape: tuple
    dtype: torch.dtype
    kwattrs: Dict[str, Any]

    @classmethod
    @abstractmethod
    def _after_subscription(cls, item: Any) -> None:
        """ Method signature for subscript argument processing. """
        raise NotImplementedError

    def __getitem__(cls, item: Any) -> GenericMeta:
        """ Defer to the metaclass which calls ``cls._after_subscription()``. """
        return SubscriptableMeta.__getitem__(cls, item)

    def __eq__(cls, other: Any) -> bool:
        """ If the dtypes and shapes match, they should be equal. """
        if not isinstance(other, _TensorMeta):
            return False
        if cls.shape != other.shape or cls.dtype != other.dtype:
            return False
        return True

    def __hash__(cls) -> int:
        """ Just calls __hash__ of SubscriptableMeta. """
        return super().__hash__()

    def __instancecheck__(cls, inst: Any) -> bool:
        """ Support expected behavior for ``isinstance(<tensor>, Tensor[<args>])``. """
        assert hasattr(cls, "shape")
        assert hasattr(cls, "dtype")
        assert hasattr(cls, "kwattrs")
        match = False
        if isinstance(inst, torch.Tensor):
            match = True  # In case of an empty tensor.

            # If we have ``cls.dtype``, we can be maximally precise.
            if cls.dtype and cls.dtype != inst.dtype:
                match = False

            # Handle ellipses.
            else:
                shape_match, _ = shapecheck(inst.shape, cls.shape)
                attr_match, _ = attrcheck(inst, cls.kwattrs)
                match = shape_match and attr_match

        return match


class _Tensor(metaclass=_TensorMeta):
    """ This class exists to keep the Tensor class as clean as possible. """

    NAME: str = "Tensor"
    DIM_TYPES: List[type] = TORCH_DIM_TYPES
    _TORCH_DTYPE_MAP: Dict[type, torch.dtype] = TORCH_DTYPE_MAP

    dtype: Optional[torch.dtype] = None
    shape: Optional[Tuple] = None
    kwattrs: Optional[Dict[str, Any]] = None

    def __new__(cls, *args: Tuple[Any], **kwargs: Dict[str, Any]) -> Any:
        raise TypeError("Cannot instantiate abstract class 'Tensor'.")

    @classmethod
    def get_dtype(cls, item: Any) -> Tuple[Optional[torch.dtype], ...]:
        """ Computes dtype. """
        dtype = None

        # Case where ``item`` is a dtype (``Tensor[torch.float64]``).
        if isinstance(item, torch.dtype):
            dtype = item

        elif isinstance(item, np.dtype):
            np_dtype_err = f"Invalid type argument '{item}'. "
            np_dtype_err += f"Numpy dtypes not supported for Tensor class. "
            np_dtype_err += f"Type arguments must be torch dtypes or in "
            np_dtype_err += f"'{list(cls._TORCH_DTYPE_MAP.keys())}'."
            raise TypeError(np_dtype_err)

        # Case where ``item`` is a python3 type (``Tensor[int]``).
        elif isinstance(item, type):
            generic_type = item
            if generic_type not in cls._TORCH_DTYPE_MAP:
                invalid_type_err = f"Invalid type argument '{generic_type}'. "
                invalid_type_err += "Type arguments must be in "
                invalid_type_err += f"'{list(cls._TORCH_DTYPE_MAP.keys())}'."
                raise TypeError(invalid_type_err)
            dtype = cls._TORCH_DTYPE_MAP[generic_type]

        return dtype, None

    @classmethod
    def _after_subscription(
        cls, item: Union[type, Optional[Union[int, EllipsisType]]]  # type: ignore
    ) -> None:
        """ Set class attributes based on the passed dtype/dim data. """
        dtype, shape, kwattrs, _ = parse_subscript(cls, item, torch.dtype)
        cls.dtype = dtype
        cls.shape = shape
        cls.kwattrs = kwattrs
