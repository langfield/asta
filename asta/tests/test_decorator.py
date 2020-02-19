# type: ignore
""" Test the ``asta.typechecked`` decorator. """
import torch
import pytest
import numpy as np
from asta import Array, Tensor, typechecked


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
def torch_correct_type(arr: Tensor[int]) -> Tensor[int]:
    """ Test function. """
    return arr


@typechecked
def torch_incorrect_type(arr: Tensor[float]) -> Tensor[int]:
    """ Test function. """
    return arr


@typechecked
def torch_incorrect_return_type(arr: Tensor[int]) -> Tensor[bytes]:
    """ Test function. """
    return arr


@typechecked
def torch_correct_dtype(arr: Tensor[torch.int32]) -> Tensor[torch.int32]:
    """ Test function. """
    return arr


@typechecked
def torch_incorrect_dtype(arr: Tensor[torch.uint8]) -> Tensor[torch.int64]:
    """ Test function. """
    return arr


@typechecked
def torch_incorrect_return_dtype(arr: Tensor[torch.int64]) -> Tensor[torch.int32]:
    """ Test function. """
    return arr


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
