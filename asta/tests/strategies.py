""" Custom hypothesis test strategies for asta. """
from typing import Any, Callable, List, Tuple, Dict

import torch
import hypothesis.strategies as st


@st.composite
def array_scalar_types(draw: Callable[[st.SearchStrategy], Any]) -> type:
    """ Strategy for valid numpy array scalar python3 types. """
    scalar_type: type = draw(st.sampled_from([int, bool, str, float, complex]))
    return scalar_type


@st.composite
def tensor_scalar_types(draw: Callable[[st.SearchStrategy], Any]) -> type:
    """ Strategy for valid torch tensor scalar python3 types. """
    scalar_type: type = draw(st.sampled_from([int, bool, bytes, float]))
    return scalar_type


@st.composite
def tensor_scalar_dtypes(draw: Callable[[st.SearchStrategy], Any]) -> torch.dtype:
    """ Strategy for valid torch tensor scalar dtypes. """
    dtypes: List[torch.dtype] = [
        torch.half,
        torch.float,
        torch.double,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.short,
        torch.int,
        torch.long,
        torch.bool,
    ]
    scalar_dtype: type = draw(st.sampled_from(dtypes))
    return scalar_dtype


@st.composite
def tensors(
    draw: Callable[[st.SearchStrategy], Any],
    dtype: torch.dtype,
    shape: Tuple[int, ...],
) -> torch.Tensor:
    """ Strategy for valid numpy array scalar python3 types. """
    raise NotImplementedError


def tensor_scalar_dtype_from_type(scalar_type: type) -> torch.dtype:
    """ Converter for valid torch tensor scalar dtypes. """
    torch_dtype_map: Dict[type, torch.dtype] = {
        int: torch.int32,
        float: torch.float32,
        bool: torch.bool,
        bytes: torch.uint8,
    }
    dtype = torch_dtype_map[scalar_type]
    return dtype
