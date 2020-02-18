#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Support for typing Numpy ndarrays. """
import numpy as np
from asta._array import _Array


class Array(_Array, np.ndarray):
    # TODO: Update examples.
    """
    A subclass of ``np.ndarray`` for use in type annotations.

    Example of an array with an undefined generic type and shape:
        ``Array``

    Example of an array with a defined generic type:
        ``Array[int]``

    Example of an array with a defined generic type and shape (rows):
        ``Array[int, 3]``
        ``Array[int, 3, ...]``
        ``Array[int, 3, None]``

    Examples of an array with a defined generic type and shape (cols):
        ``Array[int, None, 2]``
        ``Array[int, ..., 2]``

    Example of an array with a defined generic type and shape (rows and cols):
        ``Array[int, 3, 2]``
    """
