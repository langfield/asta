#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" A module for programmatically storing dimension sizes for annotations. """
from typing import Dict, Any
from sympy.core.symbol import Symbol

# pylint: disable=redefined-outer-name, too-few-public-methods
# pylint: disable=no-self-use, too-many-ancestors

dims: Dict[str, Any]


class Placeholder(Symbol):
    """ Placeholder for annotation dimensions. """

    def __init__(self, name: str) -> None:
        self.name = name
        self.unpacked = False

    def __repr__(self) -> str:
        """ String representation of placeholder. """
        return f"<Placeholder name: '{self.name}'>"

    def __iter__(self) -> object:
        """ Make sure instances can be unpacked. """
        self.unpacked = True
        yield self

    def __next__(self) -> None:
        """ Instances are empty iterators. """
        raise StopIteration


def __getattr__(name: str) -> Any:
    """ Yields the dims. """
    try:
        try:
            return dims[name]
        except KeyError:
            return Placeholder(name)
    except NameError:
        return Placeholder(name)


def __setattr__(name: str, value: Any) -> None:
    """ Sets the dims. """
    try:
        dims
    except NameError:
        dims = {}
    dims[name] = value