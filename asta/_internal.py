#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" A module for storing python objects for internal asta use. """
from typing import Any, Dict

# pylint: disable=redefined-outer-name

storage: Dict[str, Any]


def __getattr__(name: str) -> Any:
    """ Yields the objects. """
    try:
        try:
            return storage[name]
        except KeyError:
            return None
    except NameError:
        return None


def __setattr__(name: str, value: Any) -> None:
    """ Sets the objects. """
    try:
        storage
    except NameError:
        storage = {}
    storage[name] = value
