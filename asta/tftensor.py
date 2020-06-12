#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Support for typing TensorFlow tensors. """
import tensorflow as tf

from asta._tftensor import _TFTensor

# pylint: disable=too-few-public-methods


class TFTensor(_TFTensor, tf.Tensor):
    """
    A subclass of ``tf.Tensor`` for use in type annotations.

    Example of an tensor with an undefined generic type and shape:
        ``Tensor``

    Example of an tensor with a defined generic type:
        ``Tensor[int]``

    Example of an tensor with a defined tf dtype:
        ``Tensor[tf.int32]``

    Example of an tensor with a defined tf dtype and shape:
        ``Tensor[tf.int32, 3]``
        ``Tensor[tf.int64, 1, 2, 3]``

    Example of an tensor with wildcard dimension.
        ``Tensor[-1]``
        ``Tensor[int, 1, 2, -1, 3]``

    Example of an tensor with ellipses.
        ``Tensor[...]``
        ``Tensor[int, ...]``
        ``Tensor[int, 1, 2, ...]``
    """
