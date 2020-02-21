#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" A module for programmatically storing dimension sizes for annotations. """
from typing import Dict, Any

# pylint: disable=redefined-outer-name

dims: Dict[str, Any]


def __getattr__(name: str) -> Any:
    """ Yields the dims. """
    default = None
    try:
        try:
            return dims[name]
        except KeyError:
            return default
    except NameError:
        return default


def __setattr__(name: str, value: Any) -> None:
    """ Sets the dims. """
    try:
        dims
    except NameError:
        dims = {}
    dims[name] = value
