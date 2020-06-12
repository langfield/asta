#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Support for typing Numpy ndarrays. """
import numpy as np

from asta._array import _Array


class Array(_Array, np.ndarray):
    """
    A subclass of ``np.ndarray`` for use in type annotations.

    Example of an array with an undefined generic type and shape:
        ``Array``

    Example of an array with a defined generic type:
        ``Array[int]``

    Example of an array with a defined numpy dtype:
        ``Array[np.int32]``

    Example of an array with a defined numpy dtype and shape:
        ``Array[np.int32, 3]``
        ``Array[np.int64, 1, 2, 3]``

    Example of an array with wildcard dimension.
        ``Array[-1]``
        ``Array[int, 1, 2, -1, 3]``

    Example of an array with ellipses.
        ``Array[...]``
        ``Array[int, ...]``
        ``Array[int, 1, 2, ...]``
    """
