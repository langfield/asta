#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore
""" Test the ``asta.typechecked`` decorator. """
import os
import functools
from typing import List, Tuple

import numpy as np
import pytest
import hypothesis.extra.numpy as hnp
from hypothesis import given

from asta import Array, dims, shapes, symbols, typechecked

os.environ["ASTA_TYPECHECK"] = "1"

X = symbols.X
D = dims.D
S1 = shapes.S1
S2 = shapes.S2
S3 = shapes.S3

# pylint: disable=invalid-name, no-value-for-parameter


@typechecked
def np_correct_type(arr: Array[int]) -> Array[int]:
    """ Test function. """
    return arr


@typechecked
def np_incorrect_type(arr: Array[float]) -> Array[int]:
    """ Test function. """
    return arr


@typechecked
def np_incorrect_return_type(arr: Array[int]) -> Array[str]:
    """ Test function. """
    return arr


@typechecked
def np_correct_dtype(arr: Array[np.int64]) -> Array[np.int64]:
    """ Test function. """
    return arr


@typechecked
def np_incorrect_dtype(arr: Array[np.uint8]) -> Array[np.int64]:
    """ Test function. """
    return arr


@typechecked
def np_incorrect_return_dtype(arr: Array[np.int64]) -> Array[np.int32]:
    """ Test function. """
    return arr


@typechecked
def np_none(arr: Array[None]) -> Array[int]:
    """ Test function. """
    return arr


@typechecked
def np_none_return(arr: Array[int]) -> Array[None]:
    """ Test function. """
    return arr


@typechecked
def np_nones(arr: Array[None, None]) -> Array[int]:
    """ Test function. """
    return arr


@typechecked
def np_nones_return(arr: Array[int]) -> Array[None, None]:
    """ Test function. """
    return arr


@typechecked
def list_generic(l: List[Array[float, 1, 2, 3]]) -> Array[float, 1, 2, 3]:
    """ Test function. """
    t = functools.reduce(lambda x, y: x * y, l)
    return t


@typechecked
def tuple_generic(
    tup: Tuple[Array[float, 8, 32], Array[float, 8, 64]]
) -> Tuple[Array[float, 8, 32], Array[float, 8, 96]]:
    """ Test function. """
    a, b = tup
    return a, np.concatenate((a, b), axis=1)


@typechecked
def tuple_generic_inference(
    tup: Tuple[Array[float, 8, X], Array[float, 8, 2 * X]]
) -> Tuple[Array[float, 8, X], Array[float, 8, 3 * X]]:
    """ Test function. """
    a, b = tup
    return a, np.concatenate((a, b), axis=1)


@typechecked
def placeholder_arithmetic_1(t: Array[X + D]) -> Array[X + D]:
    """ Test function. """
    return t


@typechecked
def placeholder_arithmetic_2(t: Array[X ** 2 + D ** 2]) -> Array[X ** 2 + D ** 2]:
    """ Test function. """
    return t


@typechecked
def placeholder_repeated(t: Array[D, D, D]) -> Array[D, D, D]:
    """ Test function. """
    return t


@typechecked
def placeholder_incorrect_repeated(_t: Array[D, D, D]) -> Array[D, D, D]:
    """ Test function. """
    return np.ones((50, 50, 50))


@typechecked
def empty_subscript(t: Array) -> Array:
    """ Test function. """
    return t


@typechecked
def subscript_summation_1(_t: Array[S1 + (1, 2, 3) + S2 + S3]):
    """ Test function. """


@typechecked
def subscript_summation_2(_t: Array[S1 + (1,)]):
    """ Test function. """


@typechecked
def subscript_summation_3(_t: Array[S1 + (1 + D,)]):
    """ Test function. """


def test_np_typechecked():
    """ Test that decorator raises a TypeError when argument is wrong. """
    arr = np.zeros((1, 1))
    arr = arr.astype(int)
    np_correct_type(arr)
    np_correct_dtype(arr)
    with pytest.raises(TypeError):
        np_incorrect_type(arr)
    with pytest.raises(TypeError):
        np_incorrect_dtype(arr)
    with pytest.raises(TypeError):
        np_incorrect_return_type(arr)
    with pytest.raises(TypeError):
        np_incorrect_return_dtype(arr)
    with pytest.raises(TypeError):
        np_none(arr)
    with pytest.raises(TypeError):
        np_none_return(arr)
    with pytest.raises(TypeError):
        np_nones(arr)
    with pytest.raises(TypeError):
        np_nones_return(arr)


def test_subscriptable_generics():
    """ Test that subscriptable generic are typechecked properly. """
    good_list = [np.ones((1, 2, 3))] * 5
    bad_list = ([np.ones((1, 2, 3))] * 5) + [np.ones((1, 2, 2))]
    good_tuple = (np.ones((8, 32)), np.ones((8, 64)))
    bad_tuple = (np.ones((16, 32)), np.ones((16, 64)))
    bigger_tuple = (np.ones((8, 74)), np.ones((8, 148)))
    list_generic(good_list)
    with pytest.raises(TypeError):
        list_generic(bad_list)
    tuple_generic(good_tuple)
    tuple_generic_inference(bigger_tuple)
    with pytest.raises(TypeError):
        tuple_generic(bad_tuple)


def test_placeholder_arithmetic():
    """ Test that placeholders support arithmetic. """
    t = np.ones((16 + 32,))
    v = np.ones((256 + 1024,))
    u = np.ones((32, 32, 32))
    dims.D = 32

    placeholder_arithmetic_1(t)
    placeholder_arithmetic_2(v)
    placeholder_repeated(u)
    with pytest.raises(TypeError):
        placeholder_incorrect_repeated(u)


@given(hnp.arrays(dtype=hnp.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)))
def test_empty_subscript(t: Array) -> None:
    """ Test that plain ``Array`` accepts all tensor arguments. """
    empty_subscript(t)


def test_shape_arithmetic() -> None:
    """ Test that arithmetic on ``asta.shapes`` objects works. """
    shapes.S1 = (1, 1)
    shapes.S2 = (1, 1, 1)
    shapes.S3 = (1, 1, 1, 1, 1)
    t_1 = np.ones(shapes.S1 + (1, 2, 3) + shapes.S2 + shapes.S3)
    t_2 = np.ones(shapes.S1 + (1,))
    t_3 = np.ones(shapes.S1 + (1 + dims.D,))
    t_4 = np.ones(shapes.S1 + (1, 3, 3) + shapes.S2 + shapes.S3)
    t_5 = np.ones(shapes.S1 + (2,))
    t_6 = np.ones(shapes.S1 + (5 + dims.D,))
    t_7 = np.ones(shapes.S1 + (1, 2, 3) + shapes.S3 + shapes.S3)
    t_8 = np.ones(shapes.S2 + (1,))
    t_9 = np.ones(shapes.S3 + (1 + dims.D,))
    subscript_summation_1(t_1)
    subscript_summation_2(t_2)
    subscript_summation_3(t_3)
    with pytest.raises(TypeError):
        subscript_summation_1(t_4)
    with pytest.raises(TypeError):
        subscript_summation_2(t_5)
    with pytest.raises(TypeError):
        subscript_summation_3(t_6)
    with pytest.raises(TypeError):
        subscript_summation_1(t_7)
    with pytest.raises(TypeError):
        subscript_summation_2(t_8)
    with pytest.raises(TypeError):
        subscript_summation_3(t_9)
