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
    raw_type_ids: List[Union[str, int]] = list(np.sctypeDict.keys())
    type_ids: List[str] = [token for token in raw_type_ids if isinstance(token, str)]
    # This is an invalid input to ``np.dtype()``.
    type_ids.remove("V")
    type_id: str = draw(st.sampled_from(type_ids))
    print(f"Type id: {type_id}")
    
    # DEBUG
    type_id = "m8"

    dtype = np.dtype(type_id)
    return dtype


@st.composite
def shapes(draw: Callable[[SearchStrategy], Any]) -> Tuple[int, ...]:
    """ Strategy for array shapes. """
    shape_list = draw(st.lists(st.integers(min_value=0, max_value=1000), max_size=10))
    shape_list = [1, 1]
    if shape_list:
        assume(functools.reduce(lambda a, b: a * b, shape_list) < 10e6)
    return tuple(shape_list)
