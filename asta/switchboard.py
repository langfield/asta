""" A switchboard for importing asta modules with large dependencies. """
# pylint: disable=unused-import, reimported, invalid-name
from asta.unusable import Tensor, TFTensor
from asta.constants import _TORCH_IMPORTED, _TENSORFLOW_IMPORTED


if _TORCH_IMPORTED:
    from asta.tensor import Tensor
if _TENSORFLOW_IMPORTED:
    from asta.tftensor import TFTensor
