#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This module contains a superclass for the ``TFTensor`` type. """
from abc import abstractmethod
from typing import List, Optional, Any, Tuple, Dict, Union

import numpy as np
import tensorflow as tf

from asta.utils import get_shape_rep, shapecheck
from asta.scalar import Scalar
from asta.parser import parse_subscript
from asta.classes import SubscriptableMeta, GenericMeta
from asta.constants import (
    EllipsisType,
    TF_DIM_TYPES,
    TF_DTYPE_MAP,
)

# pylint: disable=unidiomatic-typecheck, too-few-public-methods, too-many-nested-blocks


class _TFTensorMeta(SubscriptableMeta):
    """ A meta class for the ``Tensor`` class. """

    shape: Union[tuple, tf.TensorShape]
    dtype: tf.dtypes.DType

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

    def __repr__(cls) -> str:
        """ String representation of ``Tensor`` class. """
        assert hasattr(cls, "shape")
        assert hasattr(cls, "dtype")
        shape = cls.shape

        if isinstance(shape, tf.TensorShape):
            shape = tuple(shape.as_list())

        if shape is None and cls.dtype is None:
            rep = f"<asta.Tensor>"
        elif shape is None and cls.dtype is not None:
            rep = f"<asta.Tensor[{cls.dtype}]>"
        elif shape is not None and cls.dtype is None:
            shape_rep = get_shape_rep(shape)
            rep = f"<asta.Tensor[{shape_rep}]>"
        else:
            shape_rep = get_shape_rep(shape)
            rep = f"<asta.Tensor[{cls.dtype}, {shape_rep}]>"

        return rep

    def __instancecheck__(cls, inst: Any) -> bool:
        """ Support expected behavior for ``isinstance(<tensor>, Tensor[<args>])``. """
        assert hasattr(cls, "shape")
        assert hasattr(cls, "dtype")
        match = False
        if isinstance(inst, tf.Tensor):
            match = True  # In case of an empty tensor.

            # If we have ``cls.dtype``, we can be maximally precise.
            if cls.dtype and cls.dtype != inst.dtype:
                match = False

            # Handle ellipses.
            elif cls.shape is not None:

                # Cast instance shape to a tuple.
                inst_shape = inst.shape
                if isinstance(inst_shape, tf.TensorShape):
                    inst_shape = tuple(inst_shape)

                match, _ = shapecheck(inst_shape, cls.shape)

        return match


class _TFTensor(metaclass=_TFTensorMeta):
    """ This class exists to keep the Tensor class as clean as possible. """

    NAME: str = "Tensor"
    DIM_TYPES: List[type] = TF_DIM_TYPES
    _TF_DTYPE_MAP: Dict[type, tf.dtypes.DType] = TF_DTYPE_MAP

    dtype: Optional[tf.dtypes.DType] = None
    shape: Optional[Tuple] = None

    def __new__(cls, *args: Tuple[Any], **kwargs: Dict[str, Any]) -> Any:
        raise TypeError("Cannot instantiate abstract class 'Tensor'.")

    @classmethod
    def get_dtype(cls, item: Any) -> Tuple[Optional[tf.dtypes.DType], ...]:
        """ Computes dtype. """
        dtype = None

        # Case where ``item`` is a dtype (``Tensor[tf.float64]``).
        if isinstance(item, tf.dtypes.DType):
            dtype = item

        elif isinstance(item, np.dtype):
            np_dtype_err = f"Invalid type argument '{item}'. "
            np_dtype_err += f"Numpy dtypes not supported for Tensor class. "
            np_dtype_err += f"Type arguments must be tf.dtypes.DTypes or in "
            np_dtype_err += f"'{list(cls._TF_DTYPE_MAP.keys())}'."
            raise TypeError(np_dtype_err)

        # Case where ``item`` is a python3 type (``Tensor[int]``).
        elif isinstance(item, type):
            generic_type = item
            if generic_type not in cls._TF_DTYPE_MAP:
                invalid_type_err = f"Invalid type argument '{generic_type}'. "
                invalid_type_err += "Type arguments must be in "
                invalid_type_err += f"'{list(cls._TF_DTYPE_MAP.keys())}'."
                raise TypeError(invalid_type_err)
            dtype = cls._TF_DTYPE_MAP[generic_type]

        return dtype, None

    @staticmethod
    def get_shape(item: Tuple) -> Optional[Tuple]:
        """ Compute shape from a shape tuple argument. """
        shape: Optional[Tuple] = None

        if item:
            if Scalar not in item and () not in item:
                shape = item
            elif item in [(Scalar,), ((),)]:
                shape = ()
            else:
                none_err = "Too many 'None' arguments. "
                none_err += "Use 'Array[None]' for scalar arrays."
                raise TypeError(none_err)

        # ``((1,2,3),)`` -> ``(1,2,3)``.
        if isinstance(shape, tuple) and len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]

        return shape

    @classmethod
    def _after_subscription(
        cls, item: Union[type, Optional[Union[int, EllipsisType]]]  # type: ignore
    ) -> None:
        """ Set class attributes based on the passed dtype/dim data. """
        if isinstance(item, tf.TensorShape):
            item = tuple(item)
        dtype, shape, _ = parse_subscript(cls, item, tf.dtypes.DType)
        cls.dtype = dtype
        cls.shape = shape
