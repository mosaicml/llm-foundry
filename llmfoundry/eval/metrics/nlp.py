from composer.metrics import InContextLearningMetric
import torch
from torch import Tensor

class InContextLearningMultipleChoiceBrierScore(InContextLearningMetric):
    r"""Computes accuracy for In-context learning (ICL) multiple choice (MC) tasks.

    ICL MC tasks consists of a series of questions with some number of possible choices (only one of which can be correct).
    At inference time each possible choice is given to the model as a separate input and the one for which the model assigns
    the lowest perplexity to the choice is considered the model's choice. The model is correct if it "chooses" the right answer.

    Context: `The dog is->fuzzy\nthe water is->hot\nthe tree is->`
    Continuation: `green`

    Adds metric state variables:
        correct (float): The number of instances where the prediction masked the target.
        total (float): The number of total instances that were predicted.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('brier_score_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, batch: dict, output_logits: torch.Tensor, labels: torch.Tensor):
        probabilites = []
        for batch_idx, cont_idx in enumerate(batch['continuation_indices']):
            # continuation indices refer to indices in the original input's token space
            cont_tok_logits = output_logits[batch_idx].index_select(dim=0, index=cont_idx - 1)
            # labels have been shifted left by one index, so the cont_idx needs to be shifted as well.
            cont_tok_targ = labels[batch_idx].index_select(dim=0, index=cont_idx - 1)
            mean_logit_of_targ_tok = cont_tok_logits.index_select(dim=1, index=cont_tok_targ).diagonal().mean()
            probabilites.append(torch.exp(-mean_logit_of_targ_tok)) # undo negative log prob to get unnormalized probability

        for (start, end), gold_idx in zip(batch['choice_groupings'], batch['gold_indices']):
            subset = probabilites[start:end]
            subset = torch.tensor(subset) / torch.tensor(subset).sum() # normalize probability
            tgt_prob = torch.zeros_like(subset)
            tgt_prob[gold_idx] = 1.0
            self.brier_score_sum += torch.nn.functional.mse_loss(subset, tgt_prob)
            self.total += torch.tensor(1.0)

    def compute(self):
        assert isinstance(self.brier_score_sum, Tensor)
        assert isinstance(self.total, Tensor)
        return self.brier_score_sum / self.total