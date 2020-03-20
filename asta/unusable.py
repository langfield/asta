#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Dummy classes for use when ``torch`` or ``tensorflow`` are not installed. """
from typing import Any
from abc import abstractmethod
from asta.classes import SubscriptableMeta, GenericMeta

# pylint: disable=import-outside-toplevel, unused-import, too-few-public-methods


class UnusableMeta(SubscriptableMeta):
    """ A meta class for the dummy ``Tensor`` and ``TFTensor`` classes. """

    @classmethod
    @abstractmethod
    def _after_subscription(cls, item: Any) -> None:
        """ Method signature for subscript argument processing. """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _import(cls) -> None:
        """ Attempts to import the dependency so a meaningful ImportError is shown. """
        raise NotImplementedError

    def __getitem__(cls, item: Any) -> GenericMeta:
        """ Should always raise an ImportError. """
        cls._import()
        return SubscriptableMeta.__getitem__(cls, item)

    def __eq__(cls, other: Any) -> bool:
        """ Should always raise an ImportError. """
        cls._import()
        return super().__eq__(other)

    def __repr__(cls) -> str:
        """ Should always raise an ImportError. """
        cls._import()
        return super().__repr__()

    def __instancecheck__(cls, inst: Any) -> bool:
        """ Should always raise an ImportError. """
        cls._import()
        return False


class Tensor(metaclass=UnusableMeta):
    """ A dummy class for use when ``torch`` is not installed. """

    @classmethod
    def _import(cls) -> None:
        """ Attempts to import the dependency so a meaningful ImportError is shown. """
        import torch


class TFTensor(metaclass=UnusableMeta):
    """ A dummy class for use when ``tensorflow`` is not installed. """

    @classmethod
    def _import(cls) -> None:
        """ Attempts to import the dependency so a meaningful ImportError is shown. """
        import tensorflow
