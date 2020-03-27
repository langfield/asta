#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore
""" Test the ``asta.typechecked`` decorator. """
import os
import functools
from typing import List, Tuple

import torch
import pytest
import numpy as np
from asta import Array, Tensor, typechecked, symbols, dims

os.environ["ASTA_TYPECHECK"] = "1"

X = symbols.X
P = dims.P

# pylint: disable=invalid-name


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
def placeholder_arithmetic_1(t: Tensor[X + P]) -> Tensor[X + P]:
    """ Test function. """
    return t


@typechecked
def placeholder_arithmetic_2(t: Tensor[X ** 2 + P ** 2]) -> Tensor[X ** 2 + P ** 2]:
    """ Test function. """
    return t


@typechecked
def placeholder_repeated(t: Tensor[P, P, P]) -> Tensor[P, P, P]:
    """ Test function. """
    return t


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
    list_generic(good_list)
    with pytest.raises(TypeError):
        list_generic(bad_list)
    tuple_generic(good_tuple)
    tuple_generic_inference(bigger_tuple)
    with pytest.raises(TypeError):
        tuple_generic(bad_tuple)


def test_placeholder_arithmetic():
    """ Test that placeholders support arithmetic. """
    t = torch.ones((16 + 32,))
    v = torch.ones((256 + 1024,))
    u = torch.ones((32, 32, 32))
    dims.P = 32
    print("Dims symbol map:", dims.symbol_map)

    placeholder_arithmetic_1(t)
    placeholder_arithmetic_2(v)
    placeholder_repeated(u)
