#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Implements variable dimension sizes for annotations. """
from typing import Any

from sympy import symbols

from asta.constants import PYATTRS


def __getattr__(name: str) -> Any:
    """ Yields the dims. """
    # Don't return sympy symbols for native module attributes.
    if name in PYATTRS:
        if name not in globals():
            raise AttributeError
        attr = globals()[name]
        return attr

    return symbols(name)
