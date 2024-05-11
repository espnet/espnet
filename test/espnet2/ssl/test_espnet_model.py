import pytest
import torch

from espnet2.asr.frontend.cnn import CNNFrontend
from espnet2.asr.frontend.default import DefaultFrontend

from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.asr.preencoder.subsampling import Subsampling

from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder

from espnet2.ssl.mask.hubert_mask import HubertMasker
from espnet2.ssl.loss.hubert_loss_ce import HuBERTLossCrossEntropy
from espnet2.ssl.loss.hubert_loss import HuBERTLoss

from espnet2.ssl.espnet_model import ESPnetSSLModel

@pytest.mark.parametrize("encoder_arch", [TransformerEncoder, EBranchformerEncoder])
@pytest.mark.parametrize("loss_fn", [HuBERTLossCrossEntropy, HuBERTLoss])
def test_espnet_model_wav2vec(frontend_arch, preencoder_arch, encoder_arch, loss_fn):
    frontend = CNNFrontend('group_norm', 'standard', True, [(3, 3, 2)])
    preencoder = LinearProjection(3, 2)
    masker = HubertMasker(4, 0.8, 'static', 0.0, 2, False, 0, 0.0, 'static', 0.0, 2, False, 0)
    encoder = encoder_arch(3, output_size=4, linear_units=4, num_blocks=2)
    loss = loss_fn(4, 5, 1)

    model = ESPnetSSLModel(
        frontend=frontend,
        specaug=None,
        normalize=None,
        preencoder=preencoder,
        encoder=encoder,
        masker=masker,
        losses=[loss]
    )

    inputs = dict(
        speech=torch.randn(2, 32, requires_grad=True),
        speech_lengths=torch.tensor([32, 16], dtype=torch.long),
        text=torch.randint(0, 5, [2, 15], dtype=torch.long),
        text_lengths=torch.tensor([15, 7], dtype=torch.long),
    )
    loss, *_ = model(**inputs)
    loss.backward()
