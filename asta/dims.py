#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" A module for programmatically storing dimension sizes for annotations. """
import sys
from typing import Dict, Optional, Any
from sympy import symbols
from sympy.core.symbol import Symbol

# pylint: disable=redefined-outer-name, too-few-public-methods, no-self-use

# Dummy map to trick pylint.
symbol_map: Dict[Symbol, Optional[int]] = {}


class Oxentiel:
    """ An instance of this object acts as a proxy for this module. """

    def __init__(self) -> None:
        # Set any attributes here - before initialisation (they remain normal attrs).
        self.symbol_map: Dict[Symbol, Optional[int]] = {}

        # After initialization, setting attributes is the same as setting an item.
        self.__initialized = True

    def __getattr__(self, name: str) -> Any:
        symbol = symbols(name)
        if symbol not in self.symbol_map:
            self.symbol_map[symbol] = None
        return symbol

    def __setattr__(self, name: str, value: Any) -> None:
        """ Maps attributes to values. Only if we are initialised. """
        # This test allows attributes to be set in the ``__init__()`` method.
        if "_Oxentiel__initialized" not in self.__dict__:
            super().__setattr__(name, value)
        # Any normal attributes are handled normally.
        elif isinstance(value, Oxentiel):
            raise AttributeError(
                "Can't assign object of type 'Oxentiel' as an attribute."
            )
        else:
            if not isinstance(value, int):
                raise TypeError("Value of a dim must be an integer.")
            symbol = symbols(name)
            self.symbol_map[symbol] = value


sys.modules[__name__] = Oxentiel()  # type: ignore[assignment]

r"""
This is a very sketchy hack to allow overriding ``__setattr__()`` on a module.
An alternative to this, if it turns out to be ill-advised, is to simply use the
module's ``__dict__`` as the storage object. This means we read the keys from
``globals()`` at import-time, and simply live with the fact that users can
override attributes of the module.

class ModuleProxy:
    def __init__(self, module):
        object.__setattr__(self, "module", module)
        self.symbol_map: Dict[Symbol, Optional[int]] = {}
        self.__initialized = True

    def __getattribute__(self, name):
        module = object.__getattribute__(self, "module")
        return getattr(module, name)

    def __setattr__(self, name, value):
        module = object.__getattribute__(self, "module")
        setattr(module, name, value)

sys.modules[__name__] = ModuleProxy(__import__(__name__))

# This never gets called. You can only override getters, not setters on modules.
def __setattr__(name, value) -> None:
    print("Testing.")
    if not isinstance(value, int):
        raise TypeError("Value of a dim must be an integer.")
    symbol = symbols(name)
    symbol_map[symbol] = value
"""
