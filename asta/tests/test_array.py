#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests for the 'Array' typing class. """
import hypothesis.extra.numpy as hnp
from hypothesis import given

from asta import Array

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
