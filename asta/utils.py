#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This module contains meta functionality for the ``Array`` type. """
import random
import functools
from typing import List, Tuple, Union, Any

from asta.constants import EllipsisType, torch, _TORCH_IMPORTED

# pylint: disable=too-many-boolean-expressions


def shapecheck(
    inst_shape: Tuple[int, ...],
    cls_shape: Tuple[Union[int, EllipsisType], ...],  # type: ignore[valid-type]
) -> Tuple[bool, List[Tuple[int, ...]]]:
    """ Check ``inst_shape`` is an instance of ``cls_shape``. """
    match = True
    assert isinstance(inst_shape, tuple)

    # The portions of ``inst_shape`` which correspond to each ``cls_shape`` elem.
    shape_pieces: List[Tuple[int, ...]] = []

    # Case 1: No ellipses or wildcards.
    if Ellipsis not in cls_shape and -1 not in cls_shape:
        if not wildcard_eq(cls_shape, inst_shape):
            match = False
        shape_pieces = [(elem,) for elem in inst_shape]

    # Case 2: Only an ellipsis, and instance shape is empty.
    elif cls_shape == (Ellipsis,) and inst_shape == ():
        match = True
        shape_pieces = [tuple()]

    # Case 3: Nonempty, not an ellipsis, and instance shape is empty.
    elif inst_shape == () != cls_shape:
        match = False

    # Case 4: Ellipsis, and instance shape has a zero (empty tensor).
    elif cls_shape == (Ellipsis,) and 0 in inst_shape:
        match = False

    # Case 5: Everything else.
    else:
        if is_subtuple((Ellipsis, Ellipsis), cls_shape)[0]:
            raise TypeError("Invalid shape: repeated '...'")

        # Determine if/where '...' bookends ``cls_shape``.
        left_bookend = False
        right_bookend = False
        ellipsis_positions: List[int] = []
        for i, elem in enumerate(cls_shape):
            if elem == Ellipsis:
                ellipsis_positions.append(i)

                # e.g. ``Array[..., 1, 2, 3]``.
                if i == 0:
                    left_bookend = True

                # e.g. ``Array[1, 2, 3, ...]``.
                if i == len(cls_shape) - 1:
                    right_bookend = True

        # Analogous to ``str.split(<elem>)``, we split the shape on '...'.
        frags: List[Tuple[int, ...]] = split(cls_shape, Ellipsis)

        # Index in ``cls_shape`` as we construct ``shape_pieces``.
        cls_idx = 0

        # ``ishape`` is the remaining portion of the instance shape.
        ishape = inst_shape
        for i, frag in enumerate(frags):

            # Look for ``frag`` in ``ishape``, and find starting index.
            is_sub, index = is_subtuple(frag, ishape)

            # Must have ``frag`` contained in ``ishape``.
            if not is_sub:
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

            # The subseq of ``ishape`` being substituted for ``...``.
            substituted = ishape[:index]

            # Can't match zero-size dims with ``...``.
            if 0 in substituted:
                match = False
                break

            cls_elem = cls_shape[cls_idx]
            if cls_elem == Ellipsis:
                shape_pieces.append(substituted)
                cls_idx += 1

            ifrag = ishape[index : index + len(frag)]
            for elem in ifrag:
                shape_pieces.append((elem,))
                cls_idx += 1

            new_start = index + len(frag)
            ishape = ishape[new_start:]

        # If ``cls_shape`` ends with ``...``, ensure substituted tuple has no 0s.
        if right_bookend and 0 in ishape:
            match = False

        if right_bookend:
            substituted = ishape
            cls_elem = cls_shape[cls_idx]
            if cls_elem == Ellipsis:
                shape_pieces.append(substituted)
                cls_idx += 1

        if match:
            reconstructed = functools.reduce(lambda a, b: a + b, shape_pieces)
            assert reconstructed == inst_shape
            assert len(shape_pieces) == len(cls_shape) == cls_idx

    return match, shape_pieces


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


def rand_split_shape(shape: Any) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """ Splits a shape and removes a nonempty continguous portion. """
    if _TORCH_IMPORTED and isinstance(shape, torch.Size):
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
