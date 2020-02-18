"""
PRIVATE MODULE: do not import (from) it directly.

This module contains class implementations.
"""
import types
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Any

# pylint: disable=too-few-public-methods

T = TypeVar("T")


class SubscriptableType(ABC):
    """ Abstract base class for subscriptable types. """

    @classmethod
    @abstractmethod
    def _after_subscription(cls, item: Any) -> None:
        """ Method signature for subscript argument processing. """
        raise NotImplementedError


class SubscriptableMeta(type, Generic[T]):
    """
    This metaclass will allow a type to become subscriptable.

    >>> class SomeType(metaclass=SubscriptableType):
    ...     pass
    >>> SomeTypeSub = SomeType['some args']
    >>> SomeTypeSub.__args__
    'some args'
    >>> SomeTypeSub.__origin__.__name__
    'SomeType'
    """

    __args__: Any
    __origin__: Any

    def __init_subclass__(cls) -> None:
        cls._hash = 0
        cls.__args__ = None
        cls.__origin__ = None

    def __getitem__(cls, item: Any) -> SubscriptableType:
        body = {
            **cls.__dict__,
            "__args__": item,
            "__origin__": cls,
        }
        bases = cls, *cls.__bases__
        print("Cls name is:", cls.__name__)
        result: SubscriptableType = type(cls.__name__, bases, body)  # type: ignore
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
