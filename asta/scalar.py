#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Scalar type class. """
from typing import List

from asta.constants import ARRAY_TYPES, GENERIC_TYPES, ScalarMeta

# pylint: disable=too-few-public-methods


class Scalar(metaclass=ScalarMeta):
    """ A generic scalar type class. """

    _GENERIC_TYPES: List[type] = GENERIC_TYPES
    _ARRAY_TYPES: List[type] = ARRAY_TYPES
