#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" A placeholder class for lazy-set shapes in asta annotations. """
from typing import Optional, List, Union, Tuple, Any
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol

# pylint: disable=no-self-use


class Placeholder:
    """ Placeholder for annotation dimensions. """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name
        self.unpacked = False
        self.composite = False
        self.contents: List[Union[Placeholder, Tuple[int, ...]]] = []

    def __repr__(self) -> str:
        """ String representation of placeholder. """
        content = f"[{self.contents}]" if self.contents else ""
        name = f"{self.name}" if self.name is not None else ""
        return f"#{name}{content}"

    def __iter__(self) -> object:
        """ Make sure instances can be unpacked. """
        self.unpacked = True
        yield self

    def __next__(self) -> None:
        """ Instances are empty iterators. """
        raise StopIteration

    def __add__(self, summand: Any) -> Any:
        """ Return the concatenation. """
        left_contents = self.contents if self.contents else [self]
        if isinstance(summand, tuple):
            for elem in summand:
                if not isinstance(elem, (int, Expr, Symbol)):
                    raise TypeError(
                        "Shape elements must be integers, symbols, or expressions."
                    )
            sum_contents = left_contents + [summand]
        elif isinstance(summand, Placeholder):
            right_contents = summand.contents if summand.contents else [summand]
            sum_contents = left_contents + right_contents
        else:
            operand_err = "Unsupported operand type(s) for +: "
            operand_err += f"'Placeholder' and '{type(summand)}'"
            raise TypeError(operand_err)

        # Construct the sum placeholder.
        result = Placeholder()
        result.contents = sum_contents
        result.composite = True
        return result
