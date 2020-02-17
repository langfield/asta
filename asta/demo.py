from typing import Any


class SubscriptableType(type):
    """ Generic metaclass for subscriptable type. """

    def __init_subclass__(cls) -> None:
        cls._hash = 0

    def __getitem__(cls, item: Any) -> Any:
        body = {
            **cls.__dict__,
            "__args__": item,
            "__origin__": cls,
        }
        bases = cls, *cls.__bases__
        result = type(cls.__name__, bases, body)
        return result

    def __eq__(cls, other: Any) -> bool:
        cls_args = getattr(cls, "__args__", None)
        cls_origin = getattr(cls, "__origin__", None)
        other_args = getattr(other, "__args__", None)
        other_origin = getattr(other, "__origin__", None)
        args_eq: bool = cls_args == other_args
        origins_eq: bool = cls_origin == other_origin
        is_eq: bool = args_eq and origins_eq
        return is_eq

    def __hash__(cls) -> int:
        """ TODO: implement. """
        if not getattr(cls, "_hash", None):
            cls._hash = 0
        return cls._hash


class AType(metaclass=SubscriptableType):
    pass


def function_1(x: AType[int]) -> AType[int]:
    return x


def function_2(x: AType[...]) -> AType[...]:
    return x


def function_3(x: AType[None]) -> AType[None]:
    return x


def main():
    x = 0
    isinstance(x, AType[int])
