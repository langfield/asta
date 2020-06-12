#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Functions for checking type annotations and their origin types. """
from typing import Any, Set, List, Tuple, Union

from oxentiel import Oxentiel
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.numbers import Number, Integer

import asta.dims
import asta.shapes
from asta.display import fail_uninitialized, fail_numerical_expression
from asta.constants import ALL_DIM_TYPES
from asta.placeholder import Placeholder

# pylint: disable=too-many-nested-blocks


def validate_placeholder_contents(
    contents: List[Union[Placeholder, Tuple[int, ...]]]
) -> None:
    """ Make sure all subplaceholders are not composite. """
    for elem in contents:
        # If it's a placeholder, it ought to be non-composite.
        if isinstance(elem, Placeholder):
            assert elem.name is not None
            assert len(elem.contents) == 0


def substitute(
    shape: Tuple[Any, ...], ox: Oxentiel
) -> Tuple[List[Any], Set[str], bool]:
    """ Parse a tuple shape, substituting placeholders for values. """
    dimension_sizes: List[Any] = []
    uninitialized_names: Set[str] = set()
    initialized = True

    for i, item in enumerate(shape):

        # Case 1: ``item`` is a sympy type.
        if isinstance(item, (Symbol, Expr)):
            expression = item

            # Use sympy to get a set of symbols used in expression.
            for symbol in item.free_symbols:

                # Check if any of the symbols in our list are in
                # ``asta.dims.symbol_map``.
                if symbol in asta.dims.symbol_map:
                    value = asta.dims.symbol_map[symbol]

                    # Out of those that are, we check if any have ``None``
                    # for their value.
                    if value is None:

                        # If so, we treat as before and raise an error.
                        name = symbol.name

                        # Prevents us from printing the same error message twice.
                        if name not in uninitialized_names:
                            fail_uninitialized(name, ox)
                        uninitialized_names.add(name)
                    else:
                        # Otherwise, we substitute in their values with sympy.
                        expression = expression.subs(symbol, value)

            # If this is a number (contains no symbols), it ought to be an integer.
            if isinstance(expression, Number):
                if not isinstance(expression, Integer):
                    fail_numerical_expression(shape[i], expression, ox)
                expression = int(expression)
            dimension_sizes.append(expression)

        # Case 2: ``item`` is a placeholder.
        elif isinstance(item, Placeholder):
            placeholder = item

            # What is item in case ``(*<placeholder>,) + (1,)``?
            # It will look like ``Array[<placeholder>, 1]``.
            if placeholder.name is not None:
                replacement: Union[Placeholder, Tuple[int, ...]]
                replacement = getattr(asta.shapes, placeholder.name)

                # Catch uninitialized placeholders.
                if isinstance(replacement, Placeholder):
                    initialized = False
                    name = placeholder.name
                    if name not in uninitialized_names:
                        fail_uninitialized(name, ox)
                    uninitialized_names.add(name)

                # Handle case where placeholder is unpacked in annotation.
                if placeholder.unpacked:
                    for elem in replacement:  # type: ignore[attr-defined]
                        dimension_sizes.append(elem)
                else:
                    dimension_sizes.append(replacement)
            else:
                # Treat case where placeholder has ``len(self.contents) > 1``.
                validate_placeholder_contents(placeholder.contents)
                contents = tuple(placeholder.contents)

                # Recursively call ``substitute()`` on the contents.
                sub_sizes, sub_uninit_names, sub_init = substitute(contents, ox)

                # Add in the results from recursive call.
                dimension_sizes.extend(sub_sizes)
                uninitialized_names = uninitialized_names.union(sub_uninit_names)
                initialized = initialized and sub_init

        # Case 3: ``item`` is a tuple.
        elif isinstance(item, tuple):

            # Recursively call ``substitute()`` on the tuple.
            sub_sizes, sub_uninit_names, sub_init = substitute(item, ox)

            # Add in the results from recursive call.
            dimension_sizes.extend(sub_sizes)
            uninitialized_names = uninitialized_names.union(sub_uninit_names)
            initialized = initialized and sub_init

        # Case 4: ``item`` is any other valid dimension type.
        elif isinstance(item, tuple(ALL_DIM_TYPES)):
            dimension_sizes.append(item)

        # Case 5: ``item`` has an invalid type.
        else:
            raise TypeError(f"Unsupported shape element type: '{type(item)}'")

    return dimension_sizes, uninitialized_names, initialized
