#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests for the 'Array' typing class. """
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
from hypothesis import given

from asta import Array
from asta.tests import strategies as strats

# pylint: disable=no-value-for-parameter


@given(hnp.arrays(dtype=strats.dtypes(), shape=strats.shapes()))
def test_array_passes_generic_isinstance(arr: Array) -> None:
    """ Make sure a generic numpy array is an instance of 'Array'. """
    pass
