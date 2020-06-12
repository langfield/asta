#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Prints a warning if typechecking will be silent. """
import os

from oxentiel import Oxentiel

from asta.config import get_ox
from asta.constants import Color

# pylint: disable=invalid-name

ox: Oxentiel = get_ox()
if "ASTA_TYPECHECK" in os.environ:
    ox.on = ox.on and os.environ["ASTA_TYPECHECK"] == "1"
if ox.on and not ox.print_passes:
    BORDER = "#" * 100
    BORDER = f"{Color.BOLD}{BORDER}{Color.END}"
    INNER = "-" * 100
    FILLER = "." * 100
    WARN = "ASTA TYPECHECKING IS ENABLED AND MAY CAUSE PERFORMANCE DEGRADATION"
    COLORED_WARN = f" {Color.RED}{WARN}{Color.END} "
    min_pad_size = 10
    pad_size = 100 - (len(WARN) + 2)
    side_size = max(pad_size // 2, min_pad_size)
    pad_parity = pad_size % 2 if side_size > 10 else 0
    left_padding = "=" * side_size
    right_padding = "=" * (side_size + pad_parity)
    PADDED_WARN = f"{left_padding}{COLORED_WARN}{right_padding}"

    print(BORDER)
    for _ in range(10):
        print(FILLER)
    print(INNER)
    print(PADDED_WARN)
    print(INNER)
    for _ in range(10):
        print(FILLER)
    print(BORDER)
