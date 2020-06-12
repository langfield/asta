#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore
""" Tests for the 'Tensor' typing class. """
from typing import List, Tuple

import torch
import pytest
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
from hypothesis import given, assume

from asta import Scalar, Tensor
from asta.tests import hpt
from asta.utils import rand_split_shape

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


def test_tensor_raises_on_two_scalar_shapes() -> None:
    """ ``Tensor[(),...]`` should raise a TypeError. """
    with pytest.raises(TypeError):
        _ = Tensor[Scalar, Scalar]
    with pytest.raises(TypeError):
        _ = Tensor[(), ()]


def test_tensor_passes_ints() -> None:
    """ Manual test for integer dtypes. """
    int8 = torch.ones((1, 1), dtype=torch.int8)
    int16 = torch.ones((1, 1), dtype=torch.int16)
    int32 = torch.ones((1, 1), dtype=torch.int32)
    int64 = torch.ones((1, 1), dtype=torch.int64)
    assert not isinstance(int8, Tensor[int])
    assert not isinstance(int16, Tensor[int])
    assert isinstance(int32, Tensor[int])
    assert not isinstance(int64, Tensor[int])


def test_tensor_fails_nones() -> None:
    """ Manual test for unintialized shape values. """
    t = torch.ones((1, 1), dtype=torch.int64)
    assert not isinstance(t, Tensor[None])
    assert not isinstance(t, Tensor[int, None])
    assert not isinstance(t, Tensor[float, None])
    assert not isinstance(t, Tensor[float, None, None])
    assert not isinstance(t, Tensor[float, None, None, None])
    t = torch.ones((), dtype=torch.int64)
    assert not isinstance(t, Tensor[None])
    assert not isinstance(t, Tensor[int, None])
    assert not isinstance(t, Tensor[float, None])
    assert not isinstance(t, Tensor[float, None, None])
    assert not isinstance(t, Tensor[float, None, None, None])


def test_tensor_discriminates_torch_dtypes() -> None:
    """ Another manual test for integer dtypes. """
    int32 = torch.ones((1, 1), dtype=torch.int32)
    assert not isinstance(int32, Tensor[torch.int16])
    assert isinstance(int32, Tensor[torch.int32])


def test_tensor_notype() -> None:
    """ Make sure Tensor only checks shape if type is not passed. """
    int8 = torch.ones((1, 1), dtype=torch.int8)
    assert isinstance(int8, Tensor[1, 1])
    assert not isinstance(int8, Tensor[1, 2])


def test_tensor_wildcard_fails_for_zero_sizes() -> None:
    """ A wildcard ``-1`` shouldn't match a zero-size. """
    t = torch.zeros(())
    empty_1 = torch.zeros((0,))
    empty_2 = torch.zeros((1, 2, 3, 0))
    empty_3 = torch.zeros((1, 0, 3, 4))
    assert not isinstance(t, Tensor[-1])
    assert not isinstance(empty_1, Tensor[-1])
    assert not isinstance(empty_2, Tensor[1, 2, 3, -1])
    assert not isinstance(empty_3, Tensor[1, -1, 3, 4])


def test_tensor_ellipsis_fails_for_zero_sizes() -> None:
    """ An empty array shouldn't pass for ``Tensor[...]``, etc. """
    t = torch.zeros(())
    empty_1 = torch.zeros((0,))
    empty_2 = torch.zeros((1, 2, 3, 0))
    empty_3 = torch.zeros((1, 0, 3, 4))
    assert isinstance(t, Tensor[...])
    assert not isinstance(empty_1, Tensor[...])
    assert not isinstance(empty_2, Tensor[...])
    assert not isinstance(empty_3, Tensor[...])
    assert not isinstance(empty_2, Tensor[1, ...])
    assert not isinstance(empty_3, Tensor[1, ...])
    assert not isinstance(empty_2, Tensor[1, 2, 3, ...])
    assert not isinstance(empty_3, Tensor[1, ..., 3, 4])


def test_tensor_ellipsis_passes_for_empty_subshapes() -> None:
    """ An Ellipsis should be a valid replacement for ``()``. """
    t = torch.zeros((1, 2, 3))
    assert isinstance(t, Tensor[...])
    assert isinstance(t, Tensor[1, 2, ...])
    assert isinstance(t, Tensor[1, 2, 3, ...])
    assert isinstance(t, Tensor[..., 1, 2, 3, ...])
    assert isinstance(t, Tensor[..., 1, ..., 2, ..., 3, ...])
    assert isinstance(t, Tensor[1, ..., 2, ..., 3])
    assert isinstance(t, Tensor[1, ..., 2, 3])
    assert isinstance(t, Tensor[..., 2, 3])
    assert isinstance(t, Tensor[..., 3])
    assert isinstance(t, Tensor[..., 2, ...])


@given(st.lists(elements=st.just(Scalar), min_size=2))
def test_tensor_raises_on_multiple_scalar_objects(scalar_list: List[Scalar]) -> None:
    """ ``Tensor[Scalar,...]`` should raise a TypeError. """
    scalar_tuple = tuple(scalar_list)
    with pytest.raises(TypeError):
        _ = Tensor[scalar_tuple]


@given(st.lists(elements=st.just(()), min_size=2))
def test_tensor_raises_on_multiple_empties(empties_list: List[tuple]) -> None:
    """ ``Tensor[(),...]`` should raise a TypeError. """
    empties_tuple = tuple(empties_list)
    with pytest.raises(TypeError):
        _ = Tensor[empties_tuple]


@given(hpt.tensors(dtype=hpt.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)))
def test_tensor_passes_generic_isinstance(t: Tensor) -> None:
    """ Make sure a generic numpy array is an instance of 'Tensor'. """
    assert isinstance(t, Tensor)
    assert isinstance(t, Tensor[t.dtype])
    assert isinstance(t, Tensor[(t.dtype,)])
    assert not isinstance(t, Tensor[None])
    assert not isinstance(t, Tensor[None, None])
    if t.shape:
        arg: tuple = (t.dtype,) + t.shape
        assert isinstance(t, Tensor[arg])


@given(hpt.tensors(dtype=hpt.scalar_dtypes(), shape=tuple()))
def test_tensor_handles_scalar_shapes(t: Tensor) -> None:
    """ Test that 'Tensor[Scalar/()]' matches a scalar. """
    assert isinstance(t, Tensor[()])
    assert isinstance(t, Tensor[t.dtype, ()])
    assert isinstance(t, Tensor[Scalar])
    assert isinstance(t, Tensor[t.dtype, Scalar])
    assert isinstance(t, Tensor[...])
    assert isinstance(t, Tensor[t.dtype, ...])
    assert not isinstance(t, Tensor[None])
    assert not isinstance(t, Tensor[t.dtype, None])


@given(hpt.tensors(dtype=hpt.scalar_dtypes(), shape=hnp.array_shapes(min_dims=1)))
def test_tensor_handles_nontrival_shapes(t: Tensor) -> None:
    """ Test that t with dim >= 1 is not scalar, and passes for its own shape. """
    left, right = rand_split_shape(t.shape)
    nones = tuple([None] * len(t.shape))
    assert isinstance(t, Tensor[left + (...,) + right])
    assert not isinstance(t, Tensor[Scalar])
    assert not isinstance(t, Tensor[()])
    assert not isinstance(t, Tensor[None])
    assert not isinstance(t, Tensor[nones])
    assert isinstance(t, Tensor[t.shape])
    assert isinstance(t, Tensor[...])
    assert isinstance(t, Tensor[(...,)])


@given(hpt.tensors(dtype=hpt.scalar_dtypes(), shape=hnp.array_shapes(min_dims=1)))
def test_tensor_handles_zeros_in_shape(t: Tensor) -> None:
    """ Test that t with dim >= 1 is not scalar, and passes for its own shape. """
    if t.shape:
        left, right = rand_split_shape(t.shape)
        assert isinstance(t, Tensor[left + (...,) + right])
    assert not isinstance(t, Tensor[Scalar])
    assert not isinstance(t, Tensor[()])
    assert isinstance(t, Tensor[t.shape])
    assert isinstance(t, Tensor[...])
    assert isinstance(t, Tensor[(...,)])


@given(st.data())
def test_tensor_handles_wildcard_shapes(data: st.DataObject) -> None:
    """
    We generate a (possibly empty) shape, add a few wildcards, then draw
    positive integer replacements for the wildcards, and assert that the
    replacement shape passed for the wildcard Tensor type.
    """
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
    t = data.draw(hpt.tensors(dtype=hpt.scalar_dtypes(), shape=tuple(rep_seq)))
    shape = tuple(seq)
    assert isinstance(t, Tensor[shape])


@given(st.data())
def test_tensor_fails_wild_wildcards(data: st.DataObject) -> None:
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
    t = data.draw(hpt.tensors(dtype=hpt.scalar_dtypes(), shape=tuple(rep_seq)))
    shape = tuple(seq)
    assert not isinstance(t, Tensor[shape])


@given(hpt.tensors(dtype=hpt.scalar_dtypes(), shape=hnp.array_shapes(min_dims=1)))
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
    """ Tests that an array is an instance of 'Tensor[<dtype>]'. """
    scalar_type = data.draw(hpt.scalar_types())
    dtype = hpt.dtype(scalar_type)
    t = data.draw(hpt.tensors(dtype=dtype, shape=hnp.array_shapes(min_dims=0)))
    assert isinstance(t, Tensor[scalar_type])
    assert isinstance(t, Tensor[(scalar_type,)])


@given(
    hpt.tensors(dtype=hpt.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)),
    hpt.scalar_dtypes(),
)
def test_tensor_is_not_instance_of_other_dtypes(t: Tensor, dtype: torch.dtype) -> None:
    """ Tests that an array isn't instance of 'Tensor[dtype]' for any other dtype. """
    assume(t.dtype != dtype)
    assert not isinstance(t, Tensor[dtype])
    assert not isinstance(t, Tensor[(dtype,)])


@given(
    hpt.tensors(dtype=hpt.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)),
    hpt.scalar_types(),
)
def test_tensor_is_not_instance_of_other_types(t: Tensor, scalar_type: type) -> None:
    """ Tests that an array isn't instance of 'Tensor[<type>]' for any other type. """
    dtype = hpt.dtype(scalar_type)
    assume(dtype != t.dtype)
    assert not isinstance(t, Tensor[scalar_type])
    assert not isinstance(t, Tensor[(scalar_type,)])
    if t.shape:
        assert not isinstance(t, Tensor[(dtype,) + t.shape])


@given(
    hpt.tensors(dtype=hpt.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)),
    hnp.array_shapes(min_dims=1),
)
def test_tensor_not_instance_right_type_wrong_shape(
    t: Tensor, shape: Tuple[int, ...]
) -> None:
    """ Tests that an array is an instance of 'Tensor[(<dtype>,)+shape]'. """
    assume(shape != t.shape)
    if t.shape:
        arg: tuple = (t.dtype,) + shape
        assert not isinstance(t, Tensor[arg])


@given(
    hpt.tensors(dtype=hpt.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)),
    hpt.scalar_types(),
    hnp.array_shapes(min_dims=0),
)
def test_tensor_not_instance_wrong_type_wrong_shape(
    t: Tensor, scalar_type: type, shape: Tuple[int, ...]
) -> None:
    """ Tests that an array is an instance of 'Tensor[(<dtype>,)+shape]'. """
    dtype = hpt.dtype(scalar_type)
    assume(shape != t.shape)
    assume(dtype != t.dtype)
    if t.shape:
        arg: tuple = (dtype,) + shape
        assert not isinstance(t, Tensor[arg])
