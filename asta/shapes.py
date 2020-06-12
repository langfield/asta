#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" A module for programmatically storing dimension sizes for annotations. """
import sys
from typing import Any, Dict, Tuple, Union

from asta.constants import PYATTRS, NoneType, ModuleType
from asta.placeholder import Placeholder

# pylint: disable=redefined-outer-name, too-few-public-methods
# pylint: disable=no-self-use, too-many-ancestors

FILELOADER = Any


def __getattr__(name: str) -> Any:
    """ This exists solely to trick pylint. """
    raise NotImplementedError


class Shapes:
    """ An instance of this object acts as a proxy for this module. """

    def __init__(self) -> None:
        # Set any attributes here - before initialisation (they remain normal attrs).
        self.placeholder_map: Dict[str, Union[Placeholder, Tuple[int, ...]]] = {}

        # After initialization, setting attributes is the same as setting an item.
        self.__initialized = True

    def __getattr__(
        self, name: str
    ) -> Union[  # type: ignore[valid-type]
        Placeholder, Tuple[int, ...], str, dict, NoneType, ModuleType, FILELOADER,
    ]:

        # Don't return sympy symbols for native module attributes.
        if name in PYATTRS:
            if name not in globals():
                raise AttributeError
            attr = globals()[name]
            return attr

        if name in self.placeholder_map:
            return self.placeholder_map[name]
        placeholder = Placeholder(name)
        self.placeholder_map[name] = placeholder
        return placeholder

    def __setattr__(self, name: str, value: Any) -> None:
        """ Maps attributes to values. Only if we are initialised. """
        # This test allows attributes to be set in the ``__init__()`` method.
        if "_Shapes__initialized" not in self.__dict__:
            super().__setattr__(name, value)
        else:
            if not isinstance(value, tuple):
                raise TypeError("Value of a shape must be an tuple.")
            for element in value:
                if not isinstance(element, int):
                    raise TypeError("Shape elements must be integers.")
            self.placeholder_map[name] = value


sys.modules[__name__] = Shapes()  # type: ignore[assignment]
