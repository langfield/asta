#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Supermetaclasses for ``_Array``-like asta types. """
import types
from abc import abstractmethod
from typing import TypeVar, Generic, Any, Optional, Tuple, List, Dict

import numpy as np
from asta.utils import shape_repr
from asta.config import get_ox
from asta.scalar import Scalar
from asta.constants import Printable
from oxentiel import Oxentiel

# pylint: disable=too-few-public-methods

T = TypeVar("T")


class GenericMeta(type, Generic[T]):
    """ Abstract base metaclass for subscriptable types. """

    kind: str
    shape: tuple
    dtype: Any
    kwattrs: Dict[str, Any]

    @classmethod
    @abstractmethod
    def _after_subscription(cls, item: Any) -> None:
        """ Method signature for subscript argument processing. """
        raise NotImplementedError


class SubscriptableMeta(GenericMeta):
    """
    This metaclass will allow a type to become subscriptable.

    >>> class SomeType(metaclass=GenericMeta):
    ...     pass
    >>> SomeTypeSub = SomeType['some args']
    >>> SomeTypeSub.__args__
    'some args'
    >>> SomeTypeSub.__origin__.__name__
    'SomeType'
    """

    DIM_TYPES: List[type]
    NAME: str
    OX: Oxentiel

    __args__: Any
    __origin__: Any

    def __init_subclass__(cls) -> None:
        cls._hash = 0
        cls.__args__ = None
        cls.__origin__ = None

    def __init__(cls, name: str, bases: Tuple[type, ...], attrs: Dict[str, Any]):
        """ Initializes the configuration object if it doesn't already exist. """
        super().__init__(name, bases, attrs)
        ox = get_ox()
        cls.OX = ox

    @classmethod
    @abstractmethod
    def _after_subscription(cls, item: Any) -> None:
        """ Method signature for subscript argument processing. """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_dtype(item: Any) -> Tuple[Optional[np.dtype], str]:
        """ Computes dtype. """
        raise NotImplementedError

    def __getitem__(cls, item: Any) -> GenericMeta:
        body = {
            **cls.__dict__,
            "__args__": item,
            "__origin__": cls,
        }
        bases = cls, *cls.__bases__
        result: GenericMeta = type(cls.__name__, bases, body)  # type: ignore

        if hasattr(result, "_after_subscription"):

            # Verify it is not a staticmethod.
            if isinstance(result._after_subscription, types.FunctionType):
                name = "_after_subscription"
                static_err = f"The '{name}' method should not be static."
                raise TypeError(static_err)

            result._after_subscription(item)
        return result

    def __eq__(cls, other: Any) -> bool:
        cls_args = getattr(cls, "__args__", None)
        cls_origin = getattr(cls, "__origin__", None)
        other_args = getattr(other, "__args__", None)
        other_origin = getattr(other, "__origin__", None)
        eq: bool = cls_args == other_args and cls_origin == other_origin
        return eq

    def __hash__(cls) -> int:
        if not getattr(cls, "_hash", None):
            cls._hash = hash("{}{}".format(cls.__origin__, cls.__args__))
        return cls._hash

    def __repr__(cls) -> str:
        """ String representation of class. """
        assert hasattr(cls, "shape")
        assert hasattr(cls, "dtype")
        assert hasattr(cls, "kwattrs")
        subscript: List[Any] = []
        if cls.dtype is not None:
            if cls.NAME == "Array":
                printable_dtype = Printable(f"np.{cls.dtype.name}")
                subscript.append(printable_dtype)
            else:
                subscript.append(cls.dtype)
        if cls.shape is not None:
            shape = tuple(cls.shape)
            printable_shape = Printable(f"shape={shape_repr(shape)}")
            subscript.append(printable_shape)
        if cls.kwattrs is not None:
            printable_kwattrs = Printable(f"attrs={cls.kwattrs}")
            subscript.append(printable_kwattrs)

        rep = f"<asta.{cls.NAME}{subscript}>"

        return rep

    @staticmethod
    def get_shape(item: Tuple) -> Optional[Tuple]:
        """ Compute shape from a shape tuple argument. """
        shape: Optional[Tuple] = None

        if item != tuple() and item is not None:
            if Scalar not in item and () not in item:
                shape = item
            elif item in [(Scalar,), ((),)]:
                shape = ()
            else:
                none_err = "Too many 'None' arguments. "
                none_err += "Use 'Array[None]' for scalar arrays."
                raise TypeError(none_err)

        # ``((1,2,3),)`` -> ``(1,2,3)``.
        if isinstance(shape, tuple) and len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]

        return shape
