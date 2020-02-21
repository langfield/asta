""" Custom hypothesis test strategies for torch tensors. """
from typing import Any, Callable, List, Tuple, Dict, Optional, Union

import torch
import numpy as np
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

# pylint: disable=no-value-for-parameter

Shape = Tuple[int, ...]

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
def scalar_types(draw: Callable[[st.SearchStrategy], Any]) -> type:
    """ Strategy for valid torch tensor scalar python3 types. """
    scalar_type: type = draw(st.sampled_from([int, bool, bytes, float]))
    return scalar_type


@st.composite
def scalar_dtypes(draw: Callable[[st.SearchStrategy], Any]) -> torch.dtype:
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


# pylint: disable=redefined-outer-name
@st.composite
def tensors(
    draw: Callable[[st.SearchStrategy], Any],
    dtype: Optional[Union[torch.dtype, st.SearchStrategy]] = None,
    shape: Optional[Union[int, Shape, st.SearchStrategy[Shape]]] = None,
) -> torch.Tensor:
    """ Strategy for valid numpy array scalar python3 types. """
    # Recurse if passed a strategy instead of a literal.
    if isinstance(dtype, st.SearchStrategy):
        return draw(dtype.flatmap(lambda d: tensors(d, shape)))
    if isinstance(shape, st.SearchStrategy):
        return draw(shape.flatmap(lambda s: tensors(dtype, s)))

    torch_dtype: torch.dtype
    torch_shape: Shape
    if dtype is None:
        torch_dtype = draw(scalar_dtypes())
    else:
        torch_dtype = dtype
    if shape is None:
        torch_shape = draw(hnp.array_shapes(min_dims=0))
    elif isinstance(shape, int):
        torch_shape = (shape,)
    else:
        torch_shape = shape
    np_dtype = _TORCH_NP_DTYPE_MAP[torch_dtype]
    arr = draw(hnp.arrays(dtype=np_dtype, shape=torch_shape))
    t = torch.from_numpy(arr)
    t = t.type(torch_dtype)
    return t


def dtype(scalar_type: type) -> torch.dtype:
    """ Converter for valid torch tensor scalar dtypes. """
    torch_dtype_map: Dict[type, torch.dtype] = {
        int: torch.int32,
        float: torch.float32,
        bool: torch.bool,
        bytes: torch.uint8,
    }
    torch_dtype = torch_dtype_map[scalar_type]
    return torch_dtype
