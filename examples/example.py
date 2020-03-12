""" An example of using runtime typechecking decorator. """
import os
import torch
import torch.nn.functional as F
from asta import Tensor, typechecked

os.environ["ASTA_TYPECHECK"] = "1"

# pylint: disable=unused-variable


def dangerous_kl(t_1: torch.Tensor, t_2: torch.Tensor) -> torch.Tensor:
    """ Computes the KL divergence of two FloatTensors of shape ``(1, 4)``. """
    divergence = F.kl_div(t_1, t_2, reduction="sum")
    return divergence


@typechecked
def safe_kl(t_1: Tensor[float, 1, 4], t_2: Tensor[float, 1, 4]) -> Tensor[float, ()]:
    """ Computes the KL divergence of two FloatTensors of shape ``(1, 4)``. """
    divergence = F.kl_div(t_1, t_2, reduction="sum")
    return divergence


def main() -> None:
    """ Example of runtime type/shape-checking. """

    # Proper shape ``(1, 4)``.
    action_logits = torch.FloatTensor([[1, 2, 3, 4]])
    optimal_logits = torch.FloatTensor([[0, 0, 0, 1]])
    action_distribution = F.softmax(action_logits, dim=1)
    optimal_distribution = F.softmax(optimal_logits, dim=1)

    # Both functions work correctly for proper input.
    div = dangerous_kl(action_distribution, optimal_distribution)
    div = safe_kl(action_distribution, optimal_distribution)

    # Improper shape ``(2, 4)``.
    action_logits = torch.FloatTensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    optimal_logits = torch.FloatTensor([[0, 0, 0, 1], [0, 0, 0, 1]])
    action_distribution = F.softmax(action_logits, dim=1)
    optimal_distribution = F.softmax(optimal_logits, dim=1)

    # Silent bug in ``dangerous_kl()``, which should take tensors of shape ``(1, 4)``.
    div = dangerous_kl(action_distribution, optimal_distribution)

    # TypeError raised in ``safe_kl()``, which sees an unexpected shape.
    print("A TypeError is raised in the safe version when we pass wrong inputs.")
    div = safe_kl(action_distribution, optimal_distribution)


if __name__ == "__main__":
    main()
