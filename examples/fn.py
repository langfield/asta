#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Example function for shapechecking. """
from typing import Tuple
import torch
from asta import Tensor, typechecked, dims, symbols

DIM = dims.DIM
X = symbols.X
Y = symbols.Y


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


@typechecked
def wrong_return(
    x: Tensor[float, X, X, X], y: Tensor[float, X, X, X],
) -> Tuple[Tensor[float, X, X, X + 1], Tensor[float, X, X, X]]:
    """ Returns first argument with variable shape. """
    return x, x


@typechecked
def identical_returns(
    x: Tensor[float, X, X, X], y: Tensor[float, Y, Y, Y],
) -> Tuple[Tensor[float, X, X, X], Tensor[float, X, X, X]]:
    """ Returns first argument with variable shape. """
    return x, y
