import pytest
import torch

from espnet2.universa.base.loss import masked_l1_loss, masked_mse_loss


@pytest.mark.parametrize("loss_fn", [masked_l1_loss, masked_mse_loss])
def test_missing_labels_have_zero_loss_and_gradient(loss_fn):
    prediction = torch.randn(3, requires_grad=True)
    loss = loss_fn(
        prediction, torch.full((3,), -100.0), torch.zeros(3, dtype=torch.bool)
    )
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros(3))
