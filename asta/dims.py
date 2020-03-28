#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" A module for programmatically storing dimension sizes for annotations. """
import sys
from typing import Dict, Optional, Any, Union
from sympy import symbols
from sympy.core.symbol import Symbol

# pylint: disable=redefined-outer-name, too-few-public-methods, no-self-use

# Dummy map to trick pylint.
symbol_map: Dict[Symbol, Optional[int]] = {}


def __getattr__(name: str) -> Any:
    """ This exists solely to trick pylint. """
    raise NotImplementedError


class Dimensions:
    """ An instance of this object acts as a proxy for this module. """

    def __init__(self) -> None:
        # Set any attributes here - before initialisation (they remain normal attrs).
        self.symbol_map: Dict[Symbol, Optional[int]] = {}

        # After initialization, setting attributes is the same as setting an item.
        self.__initialized = True

    def __getattr__(self, name: str) -> Union[Symbol, int]:
        symbol = symbols(name)
        if symbol not in self.symbol_map:
            self.symbol_map[symbol] = None
            return symbol

        value: Optional[int] = self.symbol_map[symbol]
        if value is None:
            return symbol
        assert isinstance(value, int)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        """ Maps attributes to values. Only if we are initialised. """
        # This test allows attributes to be set in the ``__init__()`` method.
        if "_Dimensions__initialized" not in self.__dict__:
            super().__setattr__(name, value)
        else:
            if not isinstance(value, int):
                raise TypeError("Value of a dim must be an integer.")
            symbol = symbols(name)
            self.symbol_map[symbol] = value


sys.modules[__name__] = Dimensions()  # type: ignore[assignment]
