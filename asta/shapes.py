#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" A module for programmatically storing shapes sizes for shape annotations. """
from typing import Dict, Any

# pylint: disable=redefined-outer-name

shapes: Dict[str, Any]


def __getattr__(name: str) -> Any:
    """ Yields the shapes. """
    default = 0
    try:
        try:
            return shapes[name]
        except KeyError:
            return default
    except NameError:
        return default


def __setattr__(name: str, value: Any) -> None:
    """ Sets the shapes. """
    try:
        shapes
    except NameError:
        shapes = {}
    shapes[name] = value
