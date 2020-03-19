#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This module contains a superclass for the ``TFTensor`` type. """
from abc import abstractmethod
from typing import List, Optional, Any, Tuple, Dict, Union

import numpy as np
import tensorflow as tf

from asta.utils import shapecheck, attrcheck
from asta.parser import parse_subscript
from asta.classes import SubscriptableMeta, GenericMeta
from asta.constants import (
    EllipsisType,
    TF_DIM_TYPES,
    TF_DTYPE_MAP,
)

# pylint: disable=unidiomatic-typecheck, too-few-public-methods, too-many-nested-blocks


class _TFTensorMeta(SubscriptableMeta):
    """ A meta class for the ``TFTensor`` class. """

    shape: Union[tuple, tf.TensorShape]
    dtype: tf.dtypes.DType
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
        if not isinstance(other, _TFTensorMeta):
            return False
        if cls.shape != other.shape or cls.dtype != other.dtype:
            return False
        return True

    def __instancecheck__(cls, inst: Any) -> bool:
        """ Support expected behavior for ``isinstance(<tensor>, TFTensor[<args>])``. """
        assert hasattr(cls, "shape")
        assert hasattr(cls, "dtype")
        assert hasattr(cls, "kwattrs")
        match = False
        if isinstance(inst, tf.Tensor):
            match = True  # In case of an empty tensor.

            # If we have ``cls.dtype``, we can be maximally precise.
            if cls.dtype and cls.dtype != inst.dtype:
                match = False

            # Handle ellipses.
            else:

                # Cast instance shape to a tuple.
                inst_shape = inst.shape
                if isinstance(inst_shape, tf.TensorShape):
                    inst_shape = tuple(inst_shape)

                shape_match, _ = shapecheck(inst_shape, cls.shape)
                attr_match, _ = attrcheck(inst, cls.kwattrs)
                match = shape_match and attr_match

        return match


class _TFTensor(metaclass=_TFTensorMeta):
    """ This class exists to keep the TFTensor class as clean as possible. """

    NAME: str = "TFTensor"
    DIM_TYPES: List[type] = TF_DIM_TYPES
    _TF_DTYPE_MAP: Dict[type, tf.dtypes.DType] = TF_DTYPE_MAP

    dtype: Optional[tf.dtypes.DType] = None
    shape: Optional[Tuple] = None
    kwattrs: Optional[Dict[str, Any]] = None

    def __new__(cls, *args: Tuple[Any], **kwargs: Dict[str, Any]) -> Any:
        raise TypeError("Cannot instantiate abstract class 'TFTensor'.")

    @classmethod
    def get_dtype(cls, item: Any) -> Tuple[Optional[tf.dtypes.DType], ...]:
        """ Computes dtype. """
        dtype = None

        # Case where ``item`` is a dtype (``TFTensor[tf.float64]``).
        if isinstance(item, tf.dtypes.DType):
            dtype = item

        elif isinstance(item, np.dtype):
            np_dtype_err = f"Invalid type argument '{item}'. "
            np_dtype_err += f"Numpy dtypes not supported for TFTensor class. "
            np_dtype_err += f"Type arguments must be tf.dtypes.DTypes or in "
            np_dtype_err += f"'{list(cls._TF_DTYPE_MAP.keys())}'."
            raise TypeError(np_dtype_err)

        # Case where ``item`` is a python3 type (``TFTensor[int]``).
        elif isinstance(item, type):
            generic_type = item
            if generic_type not in cls._TF_DTYPE_MAP:
                invalid_type_err = f"Invalid type argument '{generic_type}'. "
                invalid_type_err += "Type arguments must be in "
                invalid_type_err += f"'{list(cls._TF_DTYPE_MAP.keys())}'."
                raise TypeError(invalid_type_err)
            dtype = cls._TF_DTYPE_MAP[generic_type]

        return dtype, None

    @classmethod
    def _after_subscription(
        cls, item: Union[type, Optional[Union[int, EllipsisType]]]  # type: ignore
    ) -> None:
        """ Set class attributes based on the passed dtype/dim data. """
        if isinstance(item, tf.TensorShape):
            item = tuple(item)
        dtype, shape, kwattrs, _ = parse_subscript(cls, item, tf.dtypes.DType)
        cls.dtype = dtype
        cls.shape = shape
        cls.kwattrs = kwattrs
