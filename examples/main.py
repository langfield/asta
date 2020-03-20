#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Example script for testing typecheck toggling. """
import torch
from asta import dims, symbols
from fn import add, product, first_argument, wrong_return, identical_returns


def main() -> None:
    """ Test asta dims functionality. """
    # Before we set ``DIM``, typecheck fails.
    ob = torch.ones((5, 5, 5))
    x = torch.ones((5, 5, 5))
    y = torch.ones((5, 5, 5))
    u = torch.ones((5, 5, 4))
    v = torch.ones((5, 5, 4))
    try:
        add(ob)
    except TypeError:
        print("TYPECHECK FAILED.")

    try:
        product(u, v)
    except TypeError:
        print("TYPECHECK FAILED AS EXPECTED.")

    # Set ``DIM`` to the correct size.
    dims.DIM = 5
    add(ob)

    product(x, y)

    x = torch.ones((5, 5, 5))
    y = torch.ones((4, 4, 4))
    first_argument(x, y)

    x = torch.ones((5, 5, 5))
    y = torch.ones((5, 5, 4))
    first_argument(x, y)

    x = torch.ones((5, 5, 4))
    y = torch.ones((5, 5, 5))
    first_argument(x, y)

    x = torch.ones((5, 5, 5))
    y = torch.ones((5, 5, 5))
    wrong_return(x, y)

    x = torch.ones((6, 6, 6))
    y = torch.ones((5, 5, 5))
    identical_returns(x, y)


if __name__ == "__main__":
    main()
