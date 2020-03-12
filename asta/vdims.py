#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Implements variable dimension sizes for annotations. """
from typing import Any

# pylint: disable=redefined-outer-name, too-few-public-methods, no-self-use


class VariablePlaceholder:
    """ Placeholder for variable annotation dimensions. """

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
    return VariablePlaceholder(name)


def __setattr__(name: str, value: Any) -> None:
    """ Sets the dims. """
    raise NameError("Can't set attribute for variable dims.")
