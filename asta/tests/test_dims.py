#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore
""" Tests for valid dims attributes. """
import pytest

from asta import dims

# pylint: disable=no-value-for-parameter, invalid-name


def test_dims_rejects_tuple_attributes() -> None:
    """ Make sure sympy dims and expressions work as intended. """
    with pytest.raises(TypeError):
        dims.X = (1,)
