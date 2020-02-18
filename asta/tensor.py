#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Support for typing PyTorch tensor. """
import torch
from asta._tensor import _Tensor


class Tensor(_Tensor, torch.Tensor):
    # TODO: Update examples.
    """
    A subclass of ``torch.Tensor`` for use in type annotations.

    Example of an tensor with an undefined generic type and shape:
        ``Tensor``

    Example of an tensor with a defined generic type:
        ``Tensor[int]``

    Example of an tensor with a defined generic type and shape (rows):
        ``Tensor[int, 3]``
        ``Tensor[int, 3, ...]``
        ``Tensor[int, 3, None]``

    Examples of an tensor with a defined generic type and shape (cols):
        ``Tensor[int, None, 2]``
        ``Tensor[int, ..., 2]``

    Example of an tensor with a defined generic type and shape (rows and cols):
        ``Tensor[int, 3, 2]``
    """

    def __ipow__(self, other: torch.Tensor) -> torch.Tensor:
        """ In-place augmented arithmetic method for exponentiation (``**=``). """
        raise NotImplementedError
