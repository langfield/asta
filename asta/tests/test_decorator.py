# type: ignore
""" Test the ``asta.typechecked`` decorator. """
import os
import torch
import pytest
import numpy as np
from asta import Array, Tensor, typechecked

os.environ["ASTA_TYPECHECK"] = "1"


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
