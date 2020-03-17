#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore
""" Tests for the 'Array' typing class. """
import numpy as np

from asta import Array, symbols

# pylint: disable=no-value-for-parameter, invalid-name


def test_array_takes_expression_arguments() -> None:
    """ Make sure sympy symbols and expressions work as intended. """
    X = symbols.X
    Y = symbols.Y
    Z = symbols.Z
    a = np.zeros((1, 2, 3))

    # Univariate.
    assert isinstance(a, Array[X, X + 1, X + 2])
    assert isinstance(a, Array[X + 10, X + 11, X + 12])
    assert not isinstance(a, Array[X, X, X])
    assert not isinstance(a, Array[X, X + 1, X])
    assert not isinstance(a, Array[X, X + 1, X + 3])

    # Multivariate.
    assert isinstance(a, Array[X, X + 1, Y])
    assert isinstance(a, Array[X, Y, X + 2])
    assert isinstance(a, Array[X, Y, Z])
