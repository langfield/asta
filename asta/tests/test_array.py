#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests for the 'Array' typing class. """
import numpy as np
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
from hypothesis import given, assume

from asta import Array
from asta.tests import strategies as strats

# pylint: disable=no-value-for-parameter


@given(hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)))
def test_array_passes_generic_isinstance(arr: Array) -> None:
    """ Make sure a generic numpy array is an instance of 'Array'. """
    assert isinstance(arr, Array)


@given(hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)))
def test_array_isinstance_npdtype_shape(arr: Array) -> None:
    """ Tests that an array is an instance of 'Array[(<dtype>,)+shape]'. """
    arg: tuple = (arr.dtype,) + arr.shape
    assert isinstance(arr, Array[arg])


@given(hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)))
def test_array_isinstance_shape(arr: Array) -> None:
    """ Tests that an array is an instance of 'Array[shape]'. """
    assert isinstance(arr, Array[arr.shape])


@given(hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)))
def test_array_isinstance_dtype(arr: Array) -> None:
    """ Tests that an array is an instance of 'Array[<dtype>]'. """
    assert isinstance(arr, Array[arr.dtype])


@given(hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)))
def test_array_isinstance_dtype_tuple(arr: Array) -> None:
    """ Tests that an array is an instance of 'Array[(<dtype>,)]'. """
    assert isinstance(arr, Array[(arr.dtype,)])


@given(st.data())
def test_array_isinstance_scalar_type(data: st.DataObject) -> None:
    """ Tests that an array is an instance of 'Array[<dtype>]'. """
    scalar_type = data.draw(strats.array_scalar_types())
    dtype = np.dtype(scalar_type)
    arr = data.draw(hnp.arrays(dtype=dtype, shape=hnp.array_shapes(min_dims=0)))
    assert isinstance(arr, Array[scalar_type])


@given(st.data())
def test_array_isinstance_scalar_type_tuple(data: st.DataObject) -> None:
    """ Tests that an array is an instance of 'Array[<dtype>]'. """
    scalar_type = data.draw(strats.array_scalar_types())
    dtype = np.dtype(scalar_type)
    arr = data.draw(hnp.arrays(dtype=dtype, shape=hnp.array_shapes(min_dims=0)))
    assert isinstance(arr, Array[(scalar_type,)])


@given(
    hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)),
    hnp.scalar_dtypes(),
)
def test_array_is_not_instance_of_other_dtypes(arr: Array, dtype: np.dtype) -> None:
    """ Tests that an array isn't instance of 'Array[dtype]' for any other dtype. """
    assume(arr.dtype != dtype)
    assert not isinstance(arr, Array[dtype])


@given(
    hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)),
    hnp.scalar_dtypes(),
)
def test_array_not_instance_dtype_typles(arr: Array, dtype: np.dtype) -> None:
    """ Tests that an array isn't instance of 'Array[(dtype,)]' for any other dtype. """
    assume(arr.dtype != dtype)
    assert not isinstance(arr, Array[(dtype,)])


@given(
    hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)),
    strats.array_scalar_types(),
)
def test_array_is_not_instance_of_other_types(arr: Array, scalar_type: type) -> None:
    """ Tests that an array isn't instance of 'Array[<type>]' for any other type. """
    assume(np.dtype(scalar_type) != arr.dtype)
    assert not isinstance(arr, Array[scalar_type])


@given(
    hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)),
    strats.array_scalar_types(),
)
def test_array_not_instance_scalar_type_tuples(arr: Array, scalar_type: type) -> None:
    """ Tests that an array isn't instance of 'Array[(type,)]' for any other type. """
    assume(np.dtype(scalar_type) != arr.dtype)
    assert not isinstance(arr, Array[(scalar_type,)])
