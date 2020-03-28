#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" A placeholder class for lazy-set shapes in asta annotations. """

# pylint: disable=no-self-use


class Placeholder:
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
