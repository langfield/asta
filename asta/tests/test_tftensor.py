#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore
""" Tests for the 'TFTensor' typing class. """
from typing import List, Tuple

import pytest
import tensorflow as tf
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
from hypothesis import given, assume

from asta import Scalar, TFTensor
from asta.tests import htf
from asta.utils import rand_split_shape

# pylint: disable=no-value-for-parameter


def test_tensor_is_reflexive() -> None:
    """ Make sure ``TFTensor[<args>] == TFTensor[<args>]``. """
    # pylint: disable=comparison-with-itself
    assert TFTensor == TFTensor
    assert TFTensor[int] == TFTensor[int]
    assert TFTensor[float] != TFTensor[int]
    assert TFTensor[1, 2, 3] == TFTensor[1, 2, 3]
    assert TFTensor[1, 2, 3] != TFTensor[1, 2, 4]
    assert TFTensor[int, 1, 2, 3] == TFTensor[int, 1, 2, 3]
    assert TFTensor[int, 1, 2, 3] != TFTensor[bytes, 1, 2, 3]
    assert TFTensor[int, 1, 2, 3] != TFTensor[int, 1, 2, 4]


def test_tensor_fails_instantiation() -> None:
    """ ``TFTensor()`` should raise a TypeError. """
    with pytest.raises(TypeError):
        TFTensor()


def test_tensor_raises_on_two_scalar_shapes() -> None:
    """ ``TFTensor[(),...]`` should raise a TypeError. """
    with pytest.raises(TypeError):
        _ = TFTensor[Scalar, Scalar]
    with pytest.raises(TypeError):
        _ = TFTensor[(), ()]


def test_tensor_passes_ints() -> None:
    """ Manual test for integer dtypes. """
    int8 = tf.ones((1, 1), dtype=tf.int8)
    int16 = tf.ones((1, 1), dtype=tf.int16)
    int32 = tf.ones((1, 1), dtype=tf.int32)
    int64 = tf.ones((1, 1), dtype=tf.int64)
    assert not isinstance(int8, TFTensor[int])
    assert not isinstance(int16, TFTensor[int])
    assert isinstance(int32, TFTensor[int])
    assert not isinstance(int64, TFTensor[int])


def test_tensor_fails_nones() -> None:
    """ Manual test for unintialized shape values. """
    t = tf.ones((1, 1), dtype=tf.int64)
    assert not isinstance(t, TFTensor[None])
    assert not isinstance(t, TFTensor[int, None])
    assert not isinstance(t, TFTensor[float, None])
    assert not isinstance(t, TFTensor[float, None, None])
    assert not isinstance(t, TFTensor[float, None, None, None])
    t = tf.ones((), dtype=tf.int64)
    assert not isinstance(t, TFTensor[None])
    assert not isinstance(t, TFTensor[int, None])
    assert not isinstance(t, TFTensor[float, None])
    assert not isinstance(t, TFTensor[float, None, None])
    assert not isinstance(t, TFTensor[float, None, None, None])


def test_tensor_discriminates_tf_dtypes() -> None:
    """ Another manual test for integer dtypes. """
    int32 = tf.ones((1, 1), dtype=tf.int32)
    assert not isinstance(int32, TFTensor[tf.int16])
    assert isinstance(int32, TFTensor[tf.int32])


def test_tensor_notype() -> None:
    """ Make sure TFTensor only checks shape if type is not passed. """
    int8 = tf.ones((1, 1), dtype=tf.int8)
    assert isinstance(int8, TFTensor[1, 1])
    assert not isinstance(int8, TFTensor[1, 2])


def test_tensor_wildcard_fails_for_zero_sizes() -> None:
    """ A wildcard ``-1`` shouldn't match a zero-size. """
    t = tf.zeros(())
    empty_1 = tf.zeros((0,))
    empty_2 = tf.zeros((1, 2, 3, 0))
    empty_3 = tf.zeros((1, 0, 3, 4))
    assert not isinstance(t, TFTensor[-1])
    assert not isinstance(empty_1, TFTensor[-1])
    assert not isinstance(empty_2, TFTensor[1, 2, 3, -1])
    assert not isinstance(empty_3, TFTensor[1, -1, 3, 4])


def test_tensor_ellipsis_fails_for_zero_sizes() -> None:
    """ An empty array shouldn't pass for ``TFTensor[...]``, etc. """
    t = tf.zeros(())
    empty_1 = tf.zeros((0,))
    empty_2 = tf.zeros((1, 2, 3, 0))
    empty_3 = tf.zeros((1, 0, 3, 4))
    assert isinstance(t, TFTensor[...])
    assert not isinstance(empty_1, TFTensor[...])
    assert not isinstance(empty_2, TFTensor[...])
    assert not isinstance(empty_3, TFTensor[...])
    assert not isinstance(empty_2, TFTensor[1, ...])
    assert not isinstance(empty_3, TFTensor[1, ...])
    assert not isinstance(empty_2, TFTensor[1, 2, 3, ...])
    assert not isinstance(empty_3, TFTensor[1, ..., 3, 4])


def test_tensor_ellipsis_passes_for_empty_subshapes() -> None:
    """ An Ellipsis should be a valid replacement for ``()``. """
    t = tf.zeros((1, 2, 3))
    assert isinstance(t, TFTensor[...])
    assert isinstance(t, TFTensor[1, 2, ...])
    assert isinstance(t, TFTensor[1, 2, 3, ...])
    assert isinstance(t, TFTensor[..., 1, 2, 3, ...])
    assert isinstance(t, TFTensor[..., 1, ..., 2, ..., 3, ...])
    assert isinstance(t, TFTensor[1, ..., 2, ..., 3])
    assert isinstance(t, TFTensor[1, ..., 2, 3])
    assert isinstance(t, TFTensor[..., 2, 3])
    assert isinstance(t, TFTensor[..., 3])
    assert isinstance(t, TFTensor[..., 2, ...])


@given(st.lists(elements=st.just(Scalar), min_size=2))
def test_tensor_raises_on_multiple_scalar_objects(scalar_list: List[Scalar]) -> None:
    """ ``TFTensor[Scalar,...]`` should raise a TypeError. """
    scalar_tuple = tuple(scalar_list)
    with pytest.raises(TypeError):
        _ = TFTensor[scalar_tuple]


@given(st.lists(elements=st.just(()), min_size=2))
def test_tensor_raises_on_multiple_empties(empties_list: List[tuple]) -> None:
    """ ``TFTensor[(),...]`` should raise a TypeError. """
    empties_tuple = tuple(empties_list)
    with pytest.raises(TypeError):
        _ = TFTensor[empties_tuple]


@given(htf.tensors(dtype=htf.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)))
def test_tensor_passes_generic_isinstance(t: TFTensor) -> None:
    """ Make sure a generic numpy array is an instance of 'TFTensor'. """
    assert isinstance(t, TFTensor)
    assert isinstance(t, TFTensor[t.dtype])
    assert isinstance(t, TFTensor[(t.dtype,)])
    assert not isinstance(t, TFTensor[None])
    assert not isinstance(t, TFTensor[None, None])
    if t.shape:
        arg: tuple = (t.dtype,) + tuple(t.shape)
        assert isinstance(t, TFTensor[arg])


@given(htf.tensors(dtype=htf.scalar_dtypes(), shape=tuple()))
def test_tensor_handles_scalar_shapes(t: TFTensor) -> None:
    """ Test that 'TFTensor[Scalar/()]' matches a scalar. """
    assert isinstance(t, TFTensor[()])
    assert isinstance(t, TFTensor[t.dtype, ()])
    assert isinstance(t, TFTensor[Scalar])
    assert isinstance(t, TFTensor[t.dtype, Scalar])
    assert isinstance(t, TFTensor[...])
    assert isinstance(t, TFTensor[t.dtype, ...])
    assert not isinstance(t, TFTensor[None])
    assert not isinstance(t, TFTensor[t.dtype, None])


@given(htf.tensors(dtype=htf.scalar_dtypes(), shape=hnp.array_shapes(min_dims=1)))
def test_tensor_handles_nontrival_shapes(t: TFTensor) -> None:
    """ Test that t with dim >= 1 is not scalar, and passes for its own shape. """
    left, right = rand_split_shape(t.shape)
    left = tuple(left)
    right = tuple(right)
    nones = tuple([None] * len(t.shape))
    assert isinstance(t, TFTensor[left + (...,) + right])
    assert not isinstance(t, TFTensor[Scalar])
    assert not isinstance(t, TFTensor[()])
    assert not isinstance(t, TFTensor[None])
    assert not isinstance(t, TFTensor[nones])
    assert isinstance(t, TFTensor[t.shape])
    assert isinstance(t, TFTensor[...])
    assert isinstance(t, TFTensor[(...,)])


@given(htf.tensors(dtype=htf.scalar_dtypes(), shape=hnp.array_shapes(min_dims=1)))
def test_tensor_handles_zeros_in_shape(t: TFTensor) -> None:
    """ Test that t with dim >= 1 is not scalar, and passes for its own shape. """
    if t.shape:
        left, right = rand_split_shape(t.shape)
        left = tuple(left)
        right = tuple(right)
        assert isinstance(t, TFTensor[left + (...,) + right])
    assert not isinstance(t, TFTensor[Scalar])
    assert not isinstance(t, TFTensor[()])
    assert isinstance(t, TFTensor[t.shape])
    assert isinstance(t, TFTensor[...])
    assert isinstance(t, TFTensor[(...,)])


@given(st.data())
def test_tensor_handles_wildcard_shapes(data: st.DataObject) -> None:
    """
    We generate a (possibly empty) shape, add a few wildcards, then draw
    positive integer replacements for the wildcards, and assert that the
    replacement shape passed for the wildcard TFTensor type.
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
    t = data.draw(htf.tensors(dtype=htf.scalar_dtypes(), shape=tuple(rep_seq)))
    shape = tuple(seq)
    assert isinstance(t, TFTensor[shape])


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
    t = data.draw(htf.tensors(dtype=htf.scalar_dtypes(), shape=tuple(rep_seq)))
    shape = tuple(seq)
    assert not isinstance(t, TFTensor[shape])


@given(htf.tensors(dtype=htf.scalar_dtypes(), shape=hnp.array_shapes(min_dims=1)))
def test_tensor_handles_invalid_ellipsis_shapes(t: TFTensor) -> None:
    """ Test that t with dim >= 1 is not scalar, and passes for its own shape. """
    if t.shape:
        left, right = rand_split_shape(t.shape)
        with pytest.raises(TypeError):
            _ = TFTensor[left + (..., ...)]
        with pytest.raises(TypeError):
            _ = TFTensor[(..., ...) + right]
        with pytest.raises(TypeError):
            _ = TFTensor[(..., ...)]


@given(st.data())
def test_tensor_isinstance_scalar_type(data: st.DataObject) -> None:
    """ Tests that an array is an instance of 'TFTensor[<dtype>]'. """
    scalar_type = data.draw(htf.scalar_types())
    dtype = htf.dtype(scalar_type)
    t = data.draw(htf.tensors(dtype=dtype, shape=hnp.array_shapes(min_dims=0)))
    assert isinstance(t, TFTensor[scalar_type])
    assert isinstance(t, TFTensor[(scalar_type,)])


@given(
    htf.tensors(dtype=htf.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)),
    htf.scalar_dtypes(),
)
def test_tensor_is_not_instance_of_other_dtypes(
    t: TFTensor, dtype: tf.dtypes.DType
) -> None:
    """ Tests that an array isn't instance of 'TFTensor[dtype]' for any other dtype. """
    assume(t.dtype != dtype)
    assert not isinstance(t, TFTensor[dtype])
    assert not isinstance(t, TFTensor[(dtype,)])


@given(
    htf.tensors(dtype=htf.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)),
    htf.scalar_types(),
)
def test_tensor_is_not_instance_of_other_types(t: TFTensor, scalar_type: type) -> None:
    """ Tests that an array isn't instance of 'TFTensor[<type>]' for any other type. """
    dtype = htf.dtype(scalar_type)
    assume(dtype != t.dtype)
    assert not isinstance(t, TFTensor[scalar_type])
    assert not isinstance(t, TFTensor[(scalar_type,)])
    if t.shape:
        assert not isinstance(t, TFTensor[(dtype,) + tuple(t.shape)])


@given(
    htf.tensors(dtype=htf.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)),
    hnp.array_shapes(min_dims=1),
)
def test_tensor_not_instance_right_type_wrong_shape(
    t: TFTensor, shape: Tuple[int, ...]
) -> None:
    """ Tests that an array is an instance of 'TFTensor[(<dtype>,)+shape]'. """
    assume(shape != t.shape)
    if t.shape:
        arg: tuple = (t.dtype,) + shape
        assert not isinstance(t, TFTensor[arg])


@given(
    htf.tensors(dtype=htf.scalar_dtypes(), shape=hnp.array_shapes(min_dims=0)),
    htf.scalar_types(),
    hnp.array_shapes(min_dims=0),
)
def test_tensor_not_instance_wrong_type_wrong_shape(
    t: TFTensor, scalar_type: type, shape: Tuple[int, ...]
) -> None:
    """ Tests that an array is an instance of 'TFTensor[(<dtype>,)+shape]'. """
    dtype = htf.dtype(scalar_type)
    assume(shape != t.shape)
    assume(dtype != t.dtype)
    if t.shape:
        arg: tuple = (dtype,) + shape
        assert not isinstance(t, TFTensor[arg])
