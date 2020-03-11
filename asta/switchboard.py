""" A switchboard for importing asta modules with large dependencies. """
# pylint: disable=unused-import, reimported
from asta.unusable import Tensor
from asta.constants import _TORCH_IMPORTED


if _TORCH_IMPORTED:
    from asta.tensor import Tensor
