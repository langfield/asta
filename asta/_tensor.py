#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This module contains meta functionality for the ``Tensor`` type. """
from typing import List, Optional, Any, Tuple, Dict, Union

import torch

from asta.utils import is_subtuple, split, wildcard_eq
from asta.classes import SubscriptableMeta, SubscriptableType
from asta.constants import (
    EllipsisType,
    DIM_TYPES,
    TORCH_DTYPE_MAP,
)


# pylint: disable=unidiomatic-typecheck, too-few-public-methods, too-many-nested-blocks


class _TensorMeta(SubscriptableMeta):
    """ A meta class for the ``Tensor`` class. """

    shape: tuple
    dtype: torch.dtype

    def __getitem__(cls, item: Any) -> SubscriptableType:
        """ Defer to the metaclass which calls ``cls._after_subscription()``. """
        return SubscriptableMeta.__getitem__(cls, item)

    def __instancecheck__(cls, inst: Any) -> bool:
        """ Support expected behavior for ``isinstance(<tensor>, Tensor[<args>])``. """
        assert hasattr(cls, "shape")
        assert hasattr(cls, "dtype")
        match = False
        if isinstance(inst, torch.Tensor):
            match = True  # In case of an empty tensor.

            print("Cls dtype:", cls.dtype)
            print("inst dtype:", inst.dtype)

            # If we have ``cls.dtype``, we can be maximally precise.
            if cls.dtype and cls.dtype != inst.dtype:
                match = False

            # Handle ellipses.
            elif cls.shape is not None:
                if Ellipsis not in cls.shape and -1 not in cls.shape:
                    if not wildcard_eq(cls.shape, inst.shape):
                        match = False
                elif inst.shape == tuple() != cls.shape:
                    match = False
                else:
                    if is_subtuple((Ellipsis, Ellipsis), cls.shape)[0]:
                        raise TypeError("Invalid shape: repeated '...'")

                    # Determine if/where '...' bookends ``cls.shape``.
                    left_bookend = False
                    right_bookend = False
                    ellipsis_positions: List[int] = []
                    for i, elem in enumerate(cls.shape):
                        if elem == Ellipsis:

                            # e.g. ``Tensor[..., 1, 2, 3]``.
                            if i == 0:
                                left_bookend = True

                            # e.g. ``Tensor[1, 2, 3, ...]``.
                            if i == len(cls.shape) - 1:
                                right_bookend = True
                            ellipsis_positions.append(i)

                    # Analogous to ``str.split(<elem>)``, we split the shape on '...'.
                    frags: List[Tuple[int, ...]] = split(cls.shape, Ellipsis)

                    # Cut off end if '...' is there.
                    ishape = inst.shape
                    if left_bookend:
                        ishape = ishape[1:]
                    if right_bookend:
                        ishape = ishape[:-1]

                    for i, frag in enumerate(frags):
                        is_sub, index = is_subtuple(frag, ishape)

                        # Must have ``frag`` contained in ``ishape``.
                        if not is_sub:
                            match = False
                            break

                        # First fragment must start at 0 if '...' is not the first
                        # element of ``cls.shape``.
                        if i == 0 and not left_bookend and index != 0:
                            match = False
                            break

                        # Last fragement must end at (exclusive) ``len(ishape)`` if
                        # '...' is not the last element of ``cls.shape``.
                        if (
                            i == len(frags) - 1
                            and not right_bookend
                            and index + len(frag) != len(ishape)
                        ):
                            match = False
                            break

                        new_start = index + len(frag) + 1
                        ishape = ishape[new_start:]

        return match


class _Tensor(metaclass=_TensorMeta):
    """ This class exists to keep the Tensor class as clean as possible. """

    _DIM_TYPES: List[type] = DIM_TYPES
    _TORCH_DTYPE_MAP: Dict[type, torch.dtype] = TORCH_DTYPE_MAP

    dtype: Optional[torch.dtype] = None
    shape: Optional[Tuple] = None

    def __new__(cls, *args: Tuple[Any], **kwargs: Dict[str, Any]) -> Any:
        raise TypeError("Cannot instantiate abstract class 'Tensor'.")

    @classmethod
    def get_dtype(cls, item: Any) -> Optional[torch.dtype]:
        """ Computes dtype. """
        dtype = None

        # Case where ``item`` is a dtype (``Tensor[torch.float64]``).
        if isinstance(item, torch.dtype):
            dtype = item

        # Case where ``item`` is a python3 type (``Tensor[int]``).
        elif isinstance(item, type):
            generic_type = item
            if generic_type not in cls._TORCH_DTYPE_MAP:
                invalid_type_err = f"Invalid type argument '{generic_type}'. "
                invalid_type_err += "Type arguments must be in "
                invalid_type_err += f"'{list(cls._TORCH_DTYPE_MAP.keys())}'."
                raise TypeError(invalid_type_err)
            dtype = cls._TORCH_DTYPE_MAP[generic_type]

        return dtype

    @staticmethod
    def get_shape(item: Tuple) -> Optional[Tuple]:
        """ Compute shape from a shape tuple argument. """
        shape: Optional[Tuple] = None
        if item:
            if None not in item:
                shape = item
            elif item == (None,):
                shape = ()
            else:
                none_err = "Too many 'None' arguments. "
                none_err += "Use 'Tensor[None]' for scalar arrays."
                raise TypeError(none_err)

        return shape

    @classmethod
    def _after_subscription(
        cls, item: Union[type, Optional[Union[int, EllipsisType]]],  # type: ignore
    ) -> None:
        """ Set class attributes based on the passed dtype/dim data. """

        err = f"Invalid dimension '{item}' of type '{type(item)}'. "
        err += f"Valid dimension types: {cls._DIM_TYPES}"

        # Case where dtype is Any and shape is scalar.
        if item is None:
            cls.shape = ()

        elif isinstance(item, (torch.dtype, type)):
            cls.dtype = _Tensor.get_dtype(item)
            cls.shape = None

        # Case where dtype is not passed in, and there's one input.
        # i.e. ``Tensor[1]`` or ``Tensor[None]`` or ``Tensor[...]``.
        elif not isinstance(item, tuple):
            if type(item) not in cls._DIM_TYPES:
                raise TypeError(err)
            cls.shape = (item,)

        # Case where ``item`` is a nonempty tuple.
        elif item:

            # Case where generic type is specified.
            if isinstance(item[0], (type, torch.dtype)):
                cls.dtype = _Tensor.get_dtype(item[0])
                for i, dim in enumerate(item[1:]):
                    if type(dim) not in cls._DIM_TYPES:
                        raise TypeError(err)
                cls.shape = _Tensor.get_shape(item[1:])

            # Case where generic type is unspecified.
            else:
                for i, dim in enumerate(item):
                    if type(dim) not in cls._DIM_TYPES:
                        raise TypeError(err)
                cls.shape = _Tensor.get_shape(item)
        else:
            empty_err = "Argument to 'Tensor[]' cannot be empty tuple. "
            empty_err += "Use 'Tensor[None]' to indicate a scalar."
            raise TypeError(empty_err)

        if cls.shape is not None and is_subtuple((Ellipsis, Ellipsis), cls.shape)[0]:
            raise TypeError("Invalid shape: repeated '...'")
