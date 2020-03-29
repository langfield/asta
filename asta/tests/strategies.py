#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Custom hypothesis test strategies for asta. """
from typing import Any, Callable

import hypothesis.strategies as st

# pylint: disable=no-value-for-parameter


@st.composite
def array_scalar_types(draw: Callable[[st.SearchStrategy], Any]) -> type:
    """ Strategy for valid numpy array scalar python3 types. """
    scalar_type: type = draw(st.sampled_from([int, bool, str, float, complex]))
    return scalar_type
