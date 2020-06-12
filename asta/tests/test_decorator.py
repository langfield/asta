#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore
""" Test the ``asta.typechecked`` decorator. """
import os
import functools
from typing import List, Tuple, Optional

import numpy as np
import torch
import pytest
from hypothesis import given

from asta import Array, Tensor, dims, shapes, symbols, typechecked
from asta.tests import hpt

os.environ["ASTA_TYPECHECK"] = "1"

X = symbols.X
Y = symbols.Y
Z = symbols.Z
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
def torch_correct_type(t: Tensor[int]) -> Tensor[int]:
    """ Test function. """
    return t


@typechecked
def torch_incorrect_type(t: Tensor[float]) -> Tensor[int]:
    """ Test function. """
    return t


@typechecked
def torch_incorrect_return_type(t: Tensor[int]) -> Tensor[bytes]:
    """ Test function. """
    return t


@typechecked
def torch_correct_dtype(t: Tensor[torch.int32]) -> Tensor[torch.int32]:
    """ Test function. """
    return t


@typechecked
def torch_incorrect_dtype(t: Tensor[torch.uint8]) -> Tensor[torch.int64]:
    """ Test function. """
    return t


@typechecked
def torch_incorrect_return_dtype(t: Tensor[torch.int64]) -> Tensor[torch.int32]:
    """ Test function. """
    return t


@typechecked
def torch_none(t: Tensor[None]) -> Tensor[int]:
    """ Test function. """
    return t


@typechecked
def torch_none_return(t: Tensor[int]) -> Tensor[None]:
    """ Test function. """
    return t


@typechecked
def torch_nones(t: Tensor[None, None]) -> Tensor[int]:
    """ Test function. """
    return t


@typechecked
def torch_nones_return(t: Tensor[int]) -> Tensor[None, None]:
    """ Test function. """
    return t


@typechecked
def list_generic(l: List[Tensor[float, 1, 2, 3]]) -> Tensor[float, 1, 2, 3]:
    """ Test function. """
    t = functools.reduce(lambda x, y: x * y, l)
    return t


@typechecked
def optional_generic(
    t: Optional[Tensor[float, 1, 2, 3]] = None
) -> Optional[Tensor[float, 1, 2, 3]]:
    """ Test function. """
    return t


@typechecked
def tuple_generic(
    tup: Tuple[Tensor[float, 8, 32], Tensor[float, 8, 64]]
) -> Tuple[Tensor[float, 8, 32], Tensor[float, 8, 96]]:
    """ Test function. """
    a, b = tup
    return a, torch.cat((a, b), dim=1)


@typechecked
def tuple_generic_inference(
    tup: Tuple[Tensor[float, 8, X], Tensor[float, 8, 2 * X]]
) -> Tuple[Tensor[float, 8, X], Tensor[float, 8, 3 * X]]:
    """ Test function. """
    a, b = tup
    return a, torch.cat((a, b), dim=1)


@typechecked
def placeholder_arithmetic_1(t: Tensor[X + D]) -> Tensor[X + D]:
    """ Test function. """
    return t


@typechecked
def placeholder_arithmetic_2(t: Tensor[X ** 2 + D ** 2]) -> Tensor[X ** 2 + D ** 2]:
    """ Test function. """
    return t


@typechecked
def placeholder_repeated(t: Tensor[D, D, D]) -> Tensor[D, D, D]:
    """ Test function. """
    return t


@typechecked
def placeholder_incorrect_repeated(_t: Tensor[D, D, D]) -> Tensor[D, D, D]:
    """ Test function. """
    return torch.ones((50, 50, 50))


@typechecked
def empty_subscript(t: Tensor) -> Tensor:
    """ Test function. """
    return t


@typechecked
def subscript_summation_1(_t: Tensor[S1 + (1, 2, 3) + S2 + S3]):
    """ Test function. """


@typechecked
def subscript_summation_2(_t: Tensor[S1 + (1,)]):
    """ Test function. """


@typechecked
def subscript_summation_3(_t: Tensor[S1 + (1 + D,)]):
    """ Test function. """


@typechecked
def concatenation(
    a: Array[float, (1, 2, X)], b: Array[float, (1, 2, Y)]
) -> Array[float, (1, 2, X + Y)]:
    """ Test function. """
    return np.concatenate((a, b), axis=2)


@typechecked
def negative_size_concatenation(
    a: Array[float, (1, 2, X)], b: Array[float, (1, 2, X + Y)]
) -> Array[float, (1, 2, Z)]:
    """ Test function. """
    return np.concatenate((a, b), axis=2)


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


def test_torch_typechecked():
    """ Test that decorator raises a TypeError when argument is wrong. """
    t = torch.zeros((1, 1))
    t = t.int()
    torch_correct_type(t)
    torch_correct_dtype(t)
    with pytest.raises(TypeError):
        torch_incorrect_type(t)
    with pytest.raises(TypeError):
        torch_incorrect_dtype(t)
    with pytest.raises(TypeError):
        torch_incorrect_return_type(t)
    with pytest.raises(TypeError):
        torch_incorrect_return_dtype(t)
    with pytest.raises(TypeError):
        torch_none(t)
    with pytest.raises(TypeError):
        torch_none_return(t)
    with pytest.raises(TypeError):
        torch_nones(t)
    with pytest.raises(TypeError):
        torch_nones_return(t)


def test_subscriptable_generics():
    """ Test that subscriptable generic are typechecked properly. """
    good_list = [torch.ones((1, 2, 3))] * 5
    bad_list = ([torch.ones((1, 2, 3))] * 5) + [torch.ones((1, 2, 2))]
    good_tuple = (torch.ones((8, 32)), torch.ones((8, 64)))
    bad_tuple = (torch.ones((16, 32)), torch.ones((16, 64)))
    bigger_tuple = (torch.ones((8, 74)), torch.ones((8, 148)))
    good_optional = torch.ones((1, 2, 3))
    bad_optional = torch.ones((1, 2))
    list_generic(good_list)
    with pytest.raises(TypeError):
        list_generic(bad_list)
    tuple_generic(good_tuple)
    tuple_generic_inference(bigger_tuple)
    with pytest.raises(TypeError):
        tuple_generic(bad_tuple)
    optional_generic(good_optional)
    optional_generic()
    with pytest.raises(TypeError):
        optional_generic(bad_optional)


def test_placeholder_arithmetic():
    """ Test that placeholders support arithmetic. """
    t = torch.ones((16 + 32,))
    v = torch.ones((256 + 1024,))
    u = torch.ones((32, 32, 32))
    dims.D = 32

    placeholder_arithmetic_1(t)
    placeholder_arithmetic_2(v)
    placeholder_repeated(u)
    with pytest.raises(TypeError):
        placeholder_incorrect_repeated(u)


@given(hpt.tensors())
def test_empty_subscript(t: Tensor) -> None:
    """ Test that plain ``Tensor`` accepts all tensor arguments. """
    empty_subscript(t)


def test_shape_arithmetic() -> None:
    """ Test that arithmetic on ``asta.shapes`` objects works. """
    shapes.S1 = (1, 1)
    shapes.S2 = (1, 1, 1)
    shapes.S3 = (1, 1, 1, 1, 1)
    t_1 = torch.ones(shapes.S1 + (1, 2, 3) + shapes.S2 + shapes.S3)
    t_2 = torch.ones(shapes.S1 + (1,))
    t_3 = torch.ones(shapes.S1 + (1 + dims.D,))
    t_4 = torch.ones(shapes.S1 + (1, 3, 3) + shapes.S2 + shapes.S3)
    t_5 = torch.ones(shapes.S1 + (2,))
    t_6 = torch.ones(shapes.S1 + (5 + dims.D,))
    t_7 = torch.ones(shapes.S1 + (1, 2, 3) + shapes.S3 + shapes.S3)
    t_8 = torch.ones(shapes.S2 + (1,))
    t_9 = torch.ones(shapes.S3 + (1 + dims.D,))
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


def test_symbol_arithmetic() -> None:
    """ Tests that symbols play nicely together. """
    a = np.ones((1, 2, 3))
    b = np.ones((1, 2, 4))
    c = np.ones((1, 2, 12))
    d = np.ones((1, 2, 23))
    e = np.ones((1, 2, 10))
    f = np.ones((1, 2, 4))
    concatenation(a, b)
    concatenation(c, d)
    negative_size_concatenation(e, f)
