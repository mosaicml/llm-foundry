import pytest
import torch


from llmfoundry.metrics import TokenAccuracy

@pytest.mark.parametrize('ignore_index', [-100, -200])
@pytest.mark.parametrize('vocab_size', [100])
def test_token_accuracy(ignore_index: int, vocab_size: int):
    batch_size = int(1e6)
    torchmetrics_token_acc = TokenAccuracy(ignore_index=ignore_index)
    generated_preds = torch.rand((batch_size, vocab_size))
    true_labels = torch.randint(low=0, high=vocab_size - 1, size=(batch_size,))

    # Randomly insert ignore_index into the labels
    labels_mask = torch.rand((batch_size,))
    labels_mask[labels_mask > 0.8] = 1
    labels_mask[labels_mask <= 0.8] = 0
    labels_mask = labels_mask.bool()
    true_labels[labels_mask] = ignore_index

    true_labels = true_labels.float()
    generated_preds = generated_preds.float()

    torchmetrics_token_acc.update(generated_preds, true_labels)
    final_acc = torchmetrics_token_acc.compute()

    expected_random_acc_tensor = torch.tensor(1.0 / vocab_size)
    torch.testing.assert_close(final_acc, expected_random_acc_tensor)