""" Custom hypothesis test strategies for asta. """
from typing import Any, Callable, List, Tuple, Dict, Optional

import torch
import numpy as np
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

# pylint: disable=no-value-for-parameter

_TORCH_NP_DTYPE_MAP: Dict[torch.dtype, np.dtype] = {
    torch.half: np.float16,
    torch.float: np.float32,
    torch.double: np.float64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.short: np.int16,
    torch.int: np.int32,
    torch.long: np.int64,
    torch.bool: np.bool,
}


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
    dtype: Optional[torch.dtype] = None,
    shape: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """ Strategy for valid numpy array scalar python3 types. """
    torch_dtype: torch.dtype
    torch_shape: Tuple[int, ...]
    if dtype is None:
        torch_dtype = draw(tensor_scalar_dtypes())
    else:
        torch_dtype = dtype
    if shape is None:
        torch_shape = draw(hnp.array_shapes(min_dims=0))
    else:
        torch_shape = shape
    np_dtype = _TORCH_NP_DTYPE_MAP[torch_dtype]
    arr = draw(hnp.arrays(dtype=np_dtype, shape=torch_shape))
    t = torch.from_numpy(arr)
    t = t.type(torch_dtype)
    return t


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
