""" Example script for testing typecheck toggling. """
import torch
from asta import dims
from fn import add


def main() -> None:
    """ Test asta dims functionality. """
    # Before we set ``DIM``, typecheck fails.
    ob = torch.ones((5, 5, 5))
    try:
        add(ob)
    except TypeError as _err:
        print("TYPECHECK FAILED.")

    # Set ``DIM`` to the correct size.
    dims.DIM = 5
    add(ob)


if __name__ == "__main__":
    main()
