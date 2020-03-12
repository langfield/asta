""" Example function for shapechecking. """
import torch
from asta import Tensor, typechecked, strict, dims, vdims

DIM = dims.DIM
X = vdims.X


@typechecked
def add(
    ob: Tensor[float, DIM, DIM, DIM],
    addend: Tensor[float, DIM, DIM, DIM] = torch.ones((5, 5, 5)),
) -> Tensor[float, DIM, DIM, DIM]:
    """ Sum with default argument. """
    return ob + addend


@typechecked
def product(
    x: Tensor[float, X, X, X], y: Tensor[float, X, X, X],
) -> Tensor[float, X, X, X]:
    """ Elementwise product with variable shape. """
    return x * y
