#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Hypothesis strategies for testing the 'Array' and 'Tensor' typing classes. """
import functools
from typing import Callable, Tuple, List, Any

import numpy as np
import hypothesis.strategies as st
from hypothesis import given, assume
from hypothesis.strategies import SearchStrategy


@st.composite
def dtypes(draw: Callable[[SearchStrategy], Any]) -> np.dtype:
    """ Strategy for numpy ``dtype`` objects. """
    types: List[np.dtype] = [
        np.str,
        np.bool,
        np.intc,
        np.intp,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ]
    dtype = draw(st.sampled_from(types))
    return dtype


@st.composite
def shapes(draw: Callable[[SearchStrategy], Any]) -> Tuple[int, ...]:
    """ Strategy for array shapes. """
    shape_list = draw(st.lists(st.integers(min_value=0, max_value=1000), max_size=10))
    shape_list = [1, 1]
    if shape_list:
        assume(functools.reduce(lambda a, b: a * b, shape_list) < 10e6)
    return tuple(shape_list)
