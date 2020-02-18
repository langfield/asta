# type: ignore
""" Test the ``asta.typechecked`` decorator. """
import pytest
import numpy as np
from asta import Array, typechecked


@typechecked
def identity(arr: Array[int]) -> Array[int]:
    """ Test function. """
    return arr


@typechecked
def identity2(arr) -> Array[int]:
    """ Test function. """
    return arr


def test_typechecked():
    """ Test that decorator raises a TypeError when argument is wrong. """
    arr = np.zeros((1, 1))
    assert arr.dtype == np.float64
    with pytest.raises(TypeError):
        barr = identity(arr)
    print("Arr:", arr)
    arr = arr.astype(int)
    print(arr)
    barr = identity(arr)
    print(barr)
