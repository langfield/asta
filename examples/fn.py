""" Example function for shapechecking. """
import torch
from asta import Tensor, typechecked, check, dims, symbols

DIM = dims.DIM
X = symbols.X


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


@typechecked
def first_argument(
    x: Tensor[float, X, X, X], y: Tensor[float, X, X, X],
) -> Tensor[float, X, X, X]:
    """ Returns first argument with variable shape. """
    return x
