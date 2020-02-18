#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This module contains meta functionality for the ``Array`` type. """
import random
from typing import List, Tuple, Union

import torch
from asta.constants import EllipsisType


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
    """ Determines if two shape tuples are equal, allowing wildcards (``-1``). """
    if len(shape_1) != len(shape_2):
        return False
    for elem_1, elem_2 in zip(shape_1, shape_2):
        if (
            isinstance(elem_1, int)
            and isinstance(elem_2, int)
            and (elem_1 == -1 or elem_2 == -1)
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
