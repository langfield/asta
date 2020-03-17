#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore
""" Tests for the 'Array' typing class. """
from typing import Tuple, List

import pytest
import numpy as np
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
from hypothesis import given, assume

from asta import Array, Scalar, vdims
from asta.utils import rand_split_shape
from asta.tests import strategies as strats

# pylint: disable=no-value-for-parameter


def test_array_takes_expression_arguments() -> None:
    """ Make sure sympy symbols and expressions work as intended. """
    X = vdims.X
    a = np.zeros((1, 2, 3))
    assert isinstance(a, Array[X, X + 1, X + 2])
    assert not isinstance(a, Array[X, X, X])
    assert not isinstance(a, Array[X, X + 1, X])
    assert not isinstance(a, Array[X, X + 1, X + 3])
