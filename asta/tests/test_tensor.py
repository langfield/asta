#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore
""" Tests for the 'Tensor' typing class. """
from typing import Tuple, List

import torch
import pytest
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
from hypothesis import given, assume

from asta import Tensor
from asta.utils import rand_split_shape
from asta.tests import strategies as strats
from asta.constants import NoneType

# pylint: disable=no-value-for-parameter


def test_tensor_is_reflexive() -> None:
    """ Make sure ``Tensor[<args>] == Tensor[<args>]``. """
    # pylint: disable=comparison-with-itself
    assert Tensor == Tensor
    assert Tensor[int] == Tensor[int]
    assert Tensor[float] != Tensor[int]
    assert Tensor[1, 2, 3] == Tensor[1, 2, 3]
    assert Tensor[1, 2, 3] != Tensor[1, 2, 4]
    assert Tensor[int, 1, 2, 3] == Tensor[int, 1, 2, 3]
    assert Tensor[int, 1, 2, 3] != Tensor[bytes, 1, 2, 3]
    assert Tensor[int, 1, 2, 3] != Tensor[int, 1, 2, 4]


def test_tensor_fails_instantiation() -> None:
    """ ``Tensor()`` should raise a TypeError. """
    with pytest.raises(TypeError):
        Tensor()


def test_tensor_disallows_empty_argument() -> None:
    """ ``Empty tuple should raise a TypeError. """
    with pytest.raises(TypeError):
        _ = Tensor[()]


def test_tensor_raises_on_two_nones() -> None:
    """ ``Tensor[None,...]`` should raise a TypeError. """
    with pytest.raises(TypeError):
        _ = Tensor[None, None]


@given(st.lists(elements=st.just(None), min_size=2))
def test_tensor_raises_on_multiple_nones(none_list: List[NoneType]) -> None:
    """ ``Tensor[None,...]`` should raise a TypeError. """
    none_tuple = tuple(none_list)
    with pytest.raises(TypeError):
        _ = Tensor[none_tuple]


def test_tensor_passes_ints() -> None:
    """ Manual test for integer dtypes. """
    int8 = torch.ones((1, 1), dtype=torch.int8)
    int16 = torch.ones((1, 1), dtype=torch.int16)
    int32 = torch.ones((1, 1), dtype=torch.int32)
    int64 = torch.ones((1, 1), dtype=torch.int64)
    assert not isinstance(int8, Tensor[int])
    assert not isinstance(int16, Tensor[int])
    assert not isinstance(int64, Tensor[int])
    assert isinstance(int32, Tensor[int])


def test_tensor_discriminates_np_dtypes() -> None:
    """ Another manual test for integer dtypes. """
    int32 = torch.ones((1, 1), dtype=torch.int32)
    assert not isinstance(int32, Tensor[torch.int16])
    assert isinstance(int32, Tensor[torch.int32])


def test_tensor_notype() -> None:
    """ Make sure Tensor only checks shape if type is not passed. """
    int8 = torch.ones((1, 1), dtype=torch.int8)
    assert isinstance(int8, Tensor[1, 1])
    assert not isinstance(int8, Tensor[1, 2])


@given(strats.tensors())
def test_tensor_passes_generic_isinstance(t: Tensor) -> None:
    """ Make sure a generic numpy tensor is an instance of 'Tensor'. """
    assert isinstance(t, Tensor)
    assert isinstance(t, Tensor[t.dtype])
    assert isinstance(t, Tensor[(t.dtype,)])
    if t.shape:
        arg: tuple = (t.dtype,) + t.shape
        assert isinstance(t, Tensor[arg])


@given(st.data())
def test_tensor_scalar_isinstance_none(data: st.DataObject) -> None:
    """ Test that 'Tensor[None]' matches a scalar. """
    t = data.draw(strats.tensors(shape=tuple()))
    assert isinstance(t, Tensor[None])
    assert isinstance(t, Tensor[t.dtype, None])
    assert isinstance(t, Tensor[...])
    assert isinstance(t, Tensor[t.dtype, ...])


@given(st.data())
def test_tensor_handles_nontrival_shapes(data: st.DataObject) -> None:
    """ Test that t with dim >= 1 is not scalar, and passes for its own shape. """
    shape = data.draw(hnp.array_shapes(min_dims=1))
    t = data.draw(strats.tensors(shape=shape))
    if t.shape:
        left, right = rand_split_shape(t.shape)
        assert isinstance(t, Tensor[left + (...,) + right])
    assert not isinstance(t, Tensor[None])
    assert isinstance(t, Tensor[t.shape])
    assert isinstance(t, Tensor[...])
    assert isinstance(t, Tensor[(...,)])


@given(st.data())
def test_tensor_handles_wildcard_shapes(data: st.DataObject) -> None:
    """ Test that t with dim >= 1 is not scalar, and passes for its own shape. """
    seq = list(data.draw(hnp.array_shapes(min_dims=0)))
    num_wildcards = data.draw(st.integers(min_value=1, max_value=3))
    seq.extend([-1] * num_wildcards)
    replacements = data.draw(
        st.lists(
            st.integers(min_value=1, max_value=4),
            min_size=num_wildcards,
            max_size=num_wildcards,
        )
    )
    rep_seq = []
    for dim in seq:
        if dim == -1:
            rep_seq.append(replacements.pop())
        else:
            rep_seq.append(dim)
    t = data.draw(strats.tensors(shape=tuple(rep_seq)))
    shape = tuple(seq)
    assert isinstance(t, Tensor[shape])


@given(st.data())
def test_tensor_fails_with_wildcards(data: st.DataObject) -> None:
    """ Tests that if a wildcard is removed with a non-match, isinstance fails. """
    seq = list(data.draw(hnp.array_shapes(min_dims=0)))
    num_wildcards = data.draw(st.integers(min_value=1, max_value=3))
    seq.extend([-1] * num_wildcards)
    replacements = data.draw(
        st.lists(
            st.integers(min_value=1, max_value=4),
            min_size=num_wildcards,
            max_size=num_wildcards,
        )
    )
    rep_seq = []
    wildcard_indices = []
    for i, dim in enumerate(seq):
        if dim == -1:
            wildcard_indices.append(i)
    bad_index = data.draw(st.sampled_from(wildcard_indices))

    for i, dim in enumerate(seq):
        if dim == -1:
            rep_seq.append(replacements.pop())
        else:
            rep_seq.append(dim)

    delta = data.draw(st.integers(min_value=1, max_value=6))
    seq[bad_index] = rep_seq[bad_index] + delta
    t = data.draw(strats.tensors(shape=tuple(rep_seq)))
    shape = tuple(seq)
    assert not isinstance(t, Tensor[shape])


@given(strats.tensors())
def test_tensor_handles_invalid_ellipsis_shapes(t: Tensor) -> None:
    """ Test that t with dim >= 1 is not scalar, and passes for its own shape. """
    if t.shape:
        left, right = rand_split_shape(t.shape)
        with pytest.raises(TypeError):
            _ = Tensor[left + (..., ...)]
        with pytest.raises(TypeError):
            _ = Tensor[(..., ...) + right]
        with pytest.raises(TypeError):
            _ = Tensor[(..., ...)]


@given(st.data())
def test_tensor_isinstance_scalar_type(data: st.DataObject) -> None:
    """ Tests that an tensor is an instance of 'Tensor[<dtype>]'. """
    scalar_type = data.draw(strats.tensor_scalar_types())
    dtype = strats.tensor_scalar_dtype_from_type(scalar_type=scalar_type)
    t = data.draw(strats.tensors(dtype=dtype, shape=hnp.array_shapes(min_dims=0)))
    assert isinstance(t, Tensor[scalar_type])
    assert isinstance(t, Tensor[(scalar_type,)])


@given(strats.tensors(), strats.tensor_scalar_dtypes())
def test_tensor_is_not_instance_of_other_dtypes(t: Tensor, dtype: torch.dtype) -> None:
    """ Tests that an tensor isn't instance of 'Tensor[dtype]' for any other dtype. """
    assume(t.dtype != dtype)
    assert not isinstance(t, Tensor[dtype])
    assert not isinstance(t, Tensor[(dtype,)])


@given(st.data())
def test_tensor_is_not_instance_of_other_types(data: st.DataObject) -> None:
    """ Tests that an tensor isn't instance of 'Tensor[<type>]' for any other type. """
    t = data.draw(strats.tensors())
    scalar_type = data.draw(strats.tensor_scalar_types())
    dtype = strats.tensor_scalar_dtype_from_type(scalar_type=scalar_type)
    assume(dtype != t.dtype)
    assert not isinstance(t, Tensor[scalar_type])
    assert not isinstance(t, Tensor[(scalar_type,)])
    if t.shape:
        assert not isinstance(t, Tensor[(dtype,) + t.shape])


@given(strats.tensors(), hnp.array_shapes(min_dims=1))
def test_tensor_not_instance_right_type_wrong_shape(
    t: Tensor, shape: Tuple[int, ...]
) -> None:
    """ Tests that an tensor is an instance of 'Tensor[(<dtype>,)+shape]'. """
    assume(shape != t.shape)
    if t.shape:
        arg: tuple = (t.dtype,) + shape
        assert not isinstance(t, Tensor[arg])


@given(strats.tensors(), strats.tensor_scalar_types(), hnp.array_shapes(min_dims=0))
def test_tensor_not_instance_wrong_type_wrong_shape(
    t: Tensor, scalar_type: type, shape: Tuple[int, ...]
) -> None:
    """ Tests that an tensor is an instance of 'Tensor[(<dtype>,)+shape]'. """
    dtype = strats.tensor_scalar_dtype_from_type(scalar_type=scalar_type)
    assume(shape != t.shape)
    assume(dtype != t.dtype)
    if t.shape:
        arg: tuple = (dtype,) + shape
        assert not isinstance(t, Tensor[arg])
