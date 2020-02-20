#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This module contains meta functionality for the ``Array`` type. """
import random
from typing import List, Tuple, Union

import torch
from asta.constants import EllipsisType

# pylint: disable=too-many-boolean-expressions


def shapecheck(
    inst_shape: Tuple[int, ...],
    cls_shape: Tuple[Union[int, EllipsisType], ...],  # type: ignore[valid-type]
) -> bool:
    """ Check ``inst_shape`` is an instance of ``cls_shape``. """
    match = True
    if Ellipsis not in cls_shape and -1 not in cls_shape:
        if not wildcard_eq(cls_shape, inst_shape):
            match = False
    elif cls_shape == (Ellipsis,) and inst_shape == ():
        match = True
    elif inst_shape == () != cls_shape:
        match = False
    elif cls_shape == (Ellipsis,) and 0 in inst_shape:
        match = False
    else:
        if is_subtuple((Ellipsis, Ellipsis), cls_shape)[0]:
            raise TypeError("Invalid shape: repeated '...'")

        # Determine if/where '...' bookends ``cls_shape``.
        left_bookend = False
        right_bookend = False
        ellipsis_positions: List[int] = []
        for i, elem in enumerate(cls_shape):
            if elem == Ellipsis:

                # e.g. ``Array[..., 1, 2, 3]``.
                if i == 0:
                    left_bookend = True

                # e.g. ``Array[1, 2, 3, ...]``.
                if i == len(cls_shape) - 1:
                    right_bookend = True
                ellipsis_positions.append(i)

        # Analogous to ``str.split(<elem>)``, we split the shape on '...'.
        frags: List[Tuple[int, ...]] = split(cls_shape, Ellipsis)

        ishape = inst_shape
        for i, frag in enumerate(frags):
            is_sub, index = is_subtuple(frag, ishape)

            # Must have ``frag`` contained in ``ishape``.
            if is_sub:
                # The subseq of ``ishape`` being substituted for ``...``.
                substituted = ishape[:index]

                # Can't match zero-size dims with ``...``.
                if 0 in substituted:
                    match = False
                    break
            else:
                match = False
                break

            # First fragment must start at 0 if '...' is not the first
            # element of ``cls_shape``.
            if i == 0 and not left_bookend and index != 0:
                match = False
                break

            # Last fragement must end at (exclusive) ``len(ishape)`` if
            # '...' is not the last element of ``cls_shape``.
            if (
                i == len(frags) - 1
                and not right_bookend
                and index + len(frag) != len(ishape)
            ):
                match = False
                break

            new_start = index + len(frag)
            ishape = ishape[new_start:]

        # Check the end is a valid match if ``right_bookend`` is True.
        if right_bookend and 0 in ishape:
            match = False

    return match


def is_subtuple(
    sub: Tuple[Union[int, EllipsisType], ...],  # type: ignore[valid-type]
    tup: Tuple[Union[int, EllipsisType], ...],  # type: ignore[valid-type]
) -> Tuple[bool, int]:
    """ Check for tuple inclusion, return index of first one. """
    assert isinstance(sub, tuple)
    assert isinstance(tup, tuple)
    for i in range(len(tup) - len(sub) + 1):
        if wildcard_eq(sub, tup[i : i + len(sub)]):
            return True, i
    return False, -1


def split(
    shape: Tuple[Union[int, EllipsisType], ...],  # type: ignore[valid-type]
    elem: Union[int, EllipsisType],  # type: ignore[valid-type]
) -> List[Tuple[int, ...]]:
    """ Split on an element. """
    shape_list = list(shape)
    tokens: List[str] = []
    for num in shape_list:
        if num == elem:
            tokens.append("@")
        else:
            assert isinstance(num, int)
            tokens.append(str(num))
    seq: str = ",".join(tokens)
    comma_fragments: List[str] = seq.split("@")

    result: List[Tuple[int, ...]] = []
    for frag in comma_fragments:
        if frag:
            frag_tokens = [token for token in frag.split(",") if token]
            frag_nums = [int(token) for token in frag_tokens]
            frag_tuple = tuple(frag_nums)
            result.append(frag_tuple)

    return result


def wildcard_eq(
    shape_1: Tuple[Union[int, EllipsisType], ...],  # type: ignore[valid-type]
    shape_2: Tuple[Union[int, EllipsisType], ...],  # type: ignore[valid-type]
) -> bool:
    """
    Determines if two shape tuples are equal, allowing wildcards (``-1``),
    which can take the place of an positive integer.
    """
    if len(shape_1) != len(shape_2):
        return False
    for elem_1, elem_2 in zip(shape_1, shape_2):
        if (
            isinstance(elem_1, int)
            and isinstance(elem_2, int)
            and ((elem_1 == -1 and elem_2 != 0) or (elem_2 == -1 and elem_1 != 0))
        ):
            continue
        if elem_1 != elem_2:
            return False
    return True


def rand_split_shape(
    shape: Union[Tuple[int, ...], torch.Size]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """ Splits a shape and removes a nonempty continguous portion. """
    if isinstance(shape, torch.Size):
        shape = tuple(shape)
    start = random.randrange(len(shape))
    end = random.randint(start + 1, len(shape))
    left = shape[:start]
    right = shape[end:]
    return left, right


def get_shape_rep(shape: Tuple[int, ...]) -> str:
    """ Get stripped representation of a shape. """
    if shape == ():
        rep = f"Scalar"
    elif len(shape) == 1:
        rep = f"{shape[0]}"
    else:
        rep = repr(shape).strip("(").strip(")")
    rep = rep.replace("Ellipsis", "...")

    return rep
