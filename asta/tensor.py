#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Support for typing PyTorch tensor. """
import torch
from asta._tensor import _Tensor


class Tensor(_Tensor, torch.Tensor):
    """
    A subclass of ``torch.Tensor`` for use in type annotations.

    Example of an tensor with an undefined generic type and shape:
        ``Tensor``

    Example of an tensor with a defined generic type:
        ``Tensor[int]``

    Example of an tensor with a defined torch dtype:
        ``Tensor[torch.int32]``

    Example of an tensor with a defined torch dtype and shape:
        ``Tensor[torch.int32, 3]``
        ``Tensor[torch.int64, 1, 2, 3]``

    Example of an tensor with wildcard dimension.
        ``Tensor[-1]``
        ``Tensor[int, 1, 2, -1, 3]``

    Example of an tensor with ellipses.
        ``Tensor[...]``
        ``Tensor[int, ...]``
        ``Tensor[int, 1, 2, ...]``
    """
    def __ipow__(self, other: torch.Tensor) -> torch.Tensor:
        """ In-place augmented arithmetic method for exponentiation (``**=``). """
        raise NotImplementedError
