""" Hypothesis strategies for testing the 'Array' and 'Tensor' typing classes. """
from typing import Callable, Tuple, Any

import numpy as np
import hypothesis.strategies as st
from hypothesis import given
from hypothesis.strategies import SearchStrategy


@st.composite
def dtypes(draw: Callable[[SearchStrategy], Any]) -> np.dtype:
    """ Strategy for numpy ``dtype`` objects. """
    raise NotImplementedError


@st.composite
def shapes(draw: Callable[[SearchStrategy], Any]) -> Tuple[int, ...]:
    """ Strategy for array shapes. """
    shape_list = draw(st.lists(st.integers(min_value=0)))
    return tuple(shape_list)
