#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests for the 'Array' typing class. """
from typing import Tuple, List

import pytest
import numpy as np
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
from hypothesis import given, assume
from typish._types import Ellipsis_, NoneType

from asta import Array
from asta.tests import strategies as strats

# pylint: disable=no-value-for-parameter


def test_array_fails_instantiation() -> None:
    """ ``Array()`` should raise a TypeError. """
    with pytest.raises(TypeError):
        Array()


def test_array_disallows_empty_argument() -> None:
    """ ``Empty tuple should raise a TypeError. """
    with pytest.raises(TypeError):
        _ = Array[()]


def test_array_raises_on_multiple_nones() -> None:
    """ ``Array[None,...]`` should raise a TypeError. """
    with pytest.raises(TypeError):
        _ = Array[None, None]


@given(st.lists(elements=st.just(None), min_size=2))
def test_array_raises_on_multiple_nones(none_list: List[NoneType]) -> None:
    """ ``Array[None,...]`` should raise a TypeError. """
    none_tuple = tuple(none_list)
    with pytest.raises(TypeError):
        _ = Array[none_tuple]


def test_array_passes_ints() -> None:
    """ Manual test for integer dtypes. """
    int8 = np.ones((1, 1), dtype=np.int8)
    int16 = np.ones((1, 1), dtype=np.int16)
    int32 = np.ones((1, 1), dtype=np.int32)
    int64 = np.ones((1, 1), dtype=np.int64)
    assert not isinstance(int8, Array[int])
    assert not isinstance(int16, Array[int])
    assert not isinstance(int32, Array[int])
    assert isinstance(int64, Array[int])


def test_array_discriminates_np_dtypes() -> None:
    """ Another manual test for integer dtypes. """
    int32 = np.ones((1, 1), dtype=np.int32)
    assert not isinstance(int32, Array[np.int16])
    assert isinstance(int32, Array[np.int32])


def test_array_notype() -> None:
    """ Make sure Array only checks shape if type is not passed. """
    int8 = np.ones((1, 1), dtype=np.int8)
    assert isinstance(int8, Array[1, 1])
    assert not isinstance(int8, Array[1, 2])


@given(hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)))
def test_array_passes_generic_isinstance(arr: Array) -> None:
    """ Make sure a generic numpy array is an instance of 'Array'. """
    assert isinstance(arr, Array)
    """ Tests that an array is an instance of 'Array[(<dtype>,)+shape]'. """
    if arr.shape:
        arg: tuple = (arr.dtype,) + arr.shape
        assert isinstance(arr, Array[arg])
    """ Tests that an array is an instance of 'Array[<dtype>]'. """
    assert isinstance(arr, Array[arr.dtype])
    """ Tests that an array is an instance of 'Array[(<dtype>,)]'. """
    assert isinstance(arr, Array[(arr.dtype,)])


@given(hnp.arrays(dtype=hnp.scalar_dtypes(), shape=tuple()))
def test_array_scalar_isinstance_none(arr: Array) -> None:
    """ Test that 'Array[None]' matches a scalar. """
    assert isinstance(arr, Array[None])
    assert isinstance(arr, Array[arr.dtype, None])


@given(hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hnp.array_shapes(min_dims=1)))
def test_array_scalar_not_instance_none(arr: Array) -> None:
    """ Test that arr with dim >= 1 is not an ``Array[None]``. """
    assert not isinstance(arr, Array[None])
    """ Tests that an array is an instance of 'Array[shape]'. """
    assert isinstance(arr, Array[arr.shape])


@given(st.data())
def test_array_isinstance_scalar_type(data: st.DataObject) -> None:
    """ Tests that an array is an instance of 'Array[<dtype>]'. """
    scalar_type = data.draw(strats.array_scalar_types())
    dtype = np.dtype(scalar_type)
    arr = data.draw(hnp.arrays(dtype=dtype, shape=hnp.array_shapes(min_dims=0)))
    assert isinstance(arr, Array[scalar_type])
    """ Tests that an array is an instance of 'Array[(<dtype>,)]'. """
    assert isinstance(arr, Array[(scalar_type,)])

@given(
    hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)),
    hnp.scalar_dtypes(),
)
def test_array_is_not_instance_of_other_dtypes(arr: Array, dtype: np.dtype) -> None:
    """ Tests that an array isn't instance of 'Array[dtype]' for any other dtype. """
    assume(arr.dtype != dtype)
    assert not isinstance(arr, Array[dtype])
    """ Tests that an array isn't instance of 'Array[(dtype,)]' for any other dtype. """
    assert not isinstance(arr, Array[(dtype,)])


@given(
    hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)),
    strats.array_scalar_types(),
)
def test_array_is_not_instance_of_other_types(arr: Array, scalar_type: type) -> None:
    """ Tests that an array isn't instance of 'Array[<type>]' for any other type. """
    dtype = np.dtype(scalar_type)
    assume(dtype != arr.dtype)
    assert not isinstance(arr, Array[scalar_type])
    """ Tests that an array isn't instance of 'Array[(type,)]' for any other type. """
    assert not isinstance(arr, Array[(scalar_type,)])
    """ Tests that an array is an instance of 'Array[(<dtype>,)+shape]'. """
    if arr.shape:
        arg: tuple = (dtype,) + arr.shape
        assert not isinstance(arr, Array[arg])


@given(
    hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)),
    hnp.array_shapes(min_dims=1),
)
def test_array_not_instance_right_type_wrong_shape(
    arr: Array, shape: Tuple[int, ...]
) -> None:
    """ Tests that an array is an instance of 'Array[(<dtype>,)+shape]'. """
    assume(shape != arr.shape)
    if arr.shape:
        arg: tuple = (arr.dtype,) + shape
        assert not isinstance(arr, Array[arg])


@given(
    hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)),
    strats.array_scalar_types(),
    hnp.array_shapes(min_dims=0),
)
def test_array_not_instance_wrong_type_wrong_shape(
    arr: Array, scalar_type: type, shape: Tuple[int, ...]
) -> None:
    """ Tests that an array is an instance of 'Array[(<dtype>,)+shape]'. """
    dtype = np.dtype(scalar_type)
    assume(shape != arr.shape)
    assume(dtype != arr.dtype)
    if arr.shape:
        arg: tuple = (dtype,) + shape
        assert not isinstance(arr, Array[arg])
