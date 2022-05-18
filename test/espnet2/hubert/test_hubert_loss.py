import pytest
import torch

from espnet2.asr.encoder.hubert_encoder import \
    FairseqHubertPretrainEncoder  # noqa: H301
from espnet2.hubert.hubert_loss import HubertPretrainLoss  # noqa: H301

pytest.importorskip("fairseq")


@pytest.fixture
def hubert_args():
    encoder = FairseqHubertPretrainEncoder(
        output_size=32,
        linear_units=32,
        attention_heads=4,
        num_blocks=2,
        hubert_dict="../../../test_utils/hubert_test.txt",
    )
    bs = 2
    n_cls = 10
    logit_m_list = [torch.randn(bs, n_cls + 1)]
    logit_u_list = [torch.randn(bs, n_cls + 1)]
    padding_mask = torch.tensor([[False for _ in range(20)]])
    features_pen = torch.tensor(0.0)
    enc_outputs = {
        "logit_m_list": logit_m_list,
        "logit_u_list": logit_u_list,
        "padding_mask": padding_mask,
        "features_pen": features_pen,
    }

    return encoder.encoder, enc_outputs


def test_hubert_loss_forward_backward(hubert_args):
    hloss = HubertPretrainLoss()
    hloss(*hubert_args)
