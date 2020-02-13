from typing import Any, Callable
import hypothesis.strategies as st


@st.composite
def array_scalar_types(draw: Callable[[st.SearchStrategy], Any]) -> type:
    """ Strategy for valid numpy array scalar python3 types. """
    return draw(st.sampled_from([int, bool, str, float]))
