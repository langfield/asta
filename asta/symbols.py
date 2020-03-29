#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Implements variable dimension sizes for annotations. """
from typing import Any
from sympy import symbols


def __getattr__(name: str) -> Any:
    """ Yields the dims. """
    return symbols(name)
