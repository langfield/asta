""" Example function for shapechecking. """
import torch
from asta import Tensor, typechecked, check, dims

DIM = dims.DIM


@typechecked
def add(
    ob: Tensor[float, DIM, DIM, DIM],
    addend: Tensor[float, DIM, DIM, DIM] = torch.ones((5, 5, 5)),
) -> Tensor[float, DIM, DIM, DIM]:
    """ Identity function on an RL observation. """
    return ob + addend
