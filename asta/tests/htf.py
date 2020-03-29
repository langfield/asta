#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Custom hypothesis test strategies for TensorFlow tensors. """
from typing import Any, Callable, List, Tuple, Dict, Optional, Union

import numpy as np
import tensorflow as tf
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

from asta.constants import TF_DTYPE_MAP

# pylint: disable=no-value-for-parameter

Shape = Tuple[int, ...]

_TF_NP_DTYPE_MAP: Dict[tf.dtypes.DType, np.dtype] = {
    tf.half: np.float16,
    tf.double: np.float64,
    tf.float16: np.float16,
    tf.float32: np.float32,
    tf.float64: np.float64,
    tf.complex64: np.complex64,
    tf.complex128: np.complex128,
    tf.uint8: np.uint8,
    tf.uint16: np.uint16,
    tf.uint32: np.uint32,
    tf.uint64: np.uint64,
    tf.int8: np.int8,
    tf.int16: np.int16,
    tf.int32: np.int32,
    tf.int64: np.int64,
    tf.bool: np.bool,
    tf.string: np.unicode,
}


@st.composite
def scalar_types(draw: Callable[[st.SearchStrategy], Any]) -> type:
    """ Strategy for valid tf tensor scalar python3 types. """
    scalar_type: type = draw(st.sampled_from([int, bool, bytes, float]))
    return scalar_type


@st.composite
def scalar_dtypes(draw: Callable[[st.SearchStrategy], Any]) -> tf.dtypes.DType:
    """ Strategy for valid tf tensor scalar dtypes. """
    dtypes: List[tf.dtypes.DType] = list(_TF_NP_DTYPE_MAP.keys())
    scalar_dtype: type = draw(st.sampled_from(dtypes))
    return scalar_dtype


# pylint: disable=redefined-outer-name
@st.composite
def tensors(
    draw: Callable[[st.SearchStrategy], Any],
    dtype: Optional[Union[tf.dtypes.DType, st.SearchStrategy]] = None,
    shape: Optional[Union[int, Shape, st.SearchStrategy[Shape]]] = None,
) -> tf.Tensor:
    """ Strategy for valid numpy array scalar python3 types. """
    # Recurse if passed a strategy instead of a literal.
    if isinstance(dtype, st.SearchStrategy):
        return draw(dtype.flatmap(lambda d: tensors(d, shape)))
    if isinstance(shape, st.SearchStrategy):
        return draw(shape.flatmap(lambda s: tensors(dtype, s)))

    tf_dtype: tf.dtypes.DType
    tf_shape: Shape
    if dtype is None:
        tf_dtype = draw(scalar_dtypes())
    else:
        tf_dtype = dtype
    if shape is None:
        tf_shape = draw(hnp.array_shapes(min_dims=0))
    elif isinstance(shape, int):
        tf_shape = (shape,)
    else:
        tf_shape = shape
    np_dtype = _TF_NP_DTYPE_MAP[tf_dtype]
    arr = draw(hnp.arrays(dtype=np_dtype, shape=tf_shape))
    t = tf.convert_to_tensor(arr)
    t = tf.cast(t, tf_dtype)
    return t


def dtype(scalar_type: type) -> tf.dtypes.DType:
    """ Converter for valid tf tensor scalar dtypes. """
    tf_dtype = TF_DTYPE_MAP[scalar_type]
    return tf_dtype
