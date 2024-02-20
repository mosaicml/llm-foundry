import torch
from torchmetrics import Metric

__all__ = [
    'TokenAccuracy',
]

class TokenAccuracy(Metric):
    """
    Torchmetric to compute mean accuracy at the token level for language modeling.

    Adds metric state variables:
        correct_tokens (float): The number of correct token predictions.
        total_tokens (float): The total number of tokens predicted.

    Args:
        ignore_index (int, optional): The index of tokens to ignore, typically for padding. Default: -100.
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: False.
    """

    # Ensures torchmetrics calls update only once
    full_state_update = False

    def __init__(self, ignore_index: int = -100, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_index = ignore_index
        self.add_state('correct_tokens', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total_tokens', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Updates the internal state with results from a new batch.

        Args:
            preds (~torch.Tensor): The predictions from the model, a Tensor of logits.
            target (~torch.Tensor): A Tensor of ground-truth token values.
        """
        # Convert logits to predicted token indices
        preds = torch.argmax(preds, dim=-1)

        # Ensure predictions and targets are the same shape
        assert preds.shape == target.shape

        # Create mask for non-ignored tokens
        mask = (target != self.ignore_index)
        masked_target = target[mask]
        masked_preds = preds[mask]

        # Update correct and total counts
        self.correct_tokens += torch.sum(masked_preds == masked_target)
        self.total_tokens += masked_target.numel()

    def compute(self) -> torch.Tensor:
        """
        Aggregate the state over all processes to compute the metric.

        Returns:
            The mean accuracy across all tokens as a :class:`~torch.Tensor`.
        """
        return self.correct_tokens.float() / self.total_tokens