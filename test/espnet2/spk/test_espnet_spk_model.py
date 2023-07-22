import pytest
import torch

from espnet2.asr.specaug.specaug import SpecAug
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.spk.encoder.rawnet3_encoder import RawNet3Encoder
from espnet2.spk.espnet_model import ESPnetSpeakerModel
from espnet2.spk.loss.aamsoftmax import AAMSoftmax
from espnet2.spk.pooling.chn_attn_stat_pooling import ChnAttnStatPooling
from espnet2.spk.projector.rawnet3_projector import RawNet3Projector

rawnet3_encoder = RawNet3Encoder(model_scale=8, ndim=64, sinc_stride=16)

chn_attn_stat_pooling = ChnAttnStatPooling(input_size=96)

rawnet3_projector = RawNet3Projector(input_size=192, output_size=16)

aamsoftmax_loss = AAMSoftmax(16, 108, margin=0.3, scale=15, easy_margin=False)
aamsoftmax_em_loss = AAMSoftmax(16, 108, margin=0.3, scale=15, easy_margin=True)

specaug = SpecAug(
    apply_time_warp=True,
    time_warp_window=5,
    time_warp_mode="bicubic",
    apply_freq_mask=True,
    freq_mask_width_range=[0, 30],
    num_freq_mask=2,
    apply_time_mask=True,
    time_mask_width_range=[0, 40],
    num_time_mask=2,
)

normalize = UtteranceMVN()


@pytest.mark.parametrize("encoder, projector", [(rawnet3_encoder, rawnet3_projector)])
@pytest.mark.parametrize("pooling", [chn_attn_stat_pooling])
@pytest.mark.parametrize("training", [True, False])
def test_single_channel_spk_model(encoder, pooling, projector, training):
    inputs = torch.randn(2, 8000)
    ilens = torch.LongTensor([8000, 7800])
    spk_labels = torch.randint(0, 108, (2,))
    spk_model = ESPnetSpeakerModel(
        frontend=None,
        specaug=None,
        normalize=None,
        encoder=encoder,
        pooling=pooling,
        projector=projector,
        loss=aamsoftmax_loss,
    )

    if training:
        spk_model.train()
    else:
        spk_model.eval()

    kwargs = {"speech": inputs, "speech_lengths": ilens, "spk_labels": spk_labels}
    loss, stats, weight = spk_model(**kwargs)

    if training:
        loss.backward()


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("loss", [aamsoftmax_loss, aamsoftmax_em_loss])
def test_spk_loss(training, loss):
    inputs = torch.randn(2, 8000)
    ilens = torch.LongTensor([8000, 7800])
    spk_labels = torch.randint(0, 108, (2,))
    spk_model = ESPnetSpeakerModel(
        frontend=None,
        specaug=None,
        normalize=normalize,
        encoder=rawnet3_encoder,
        pooling=chn_attn_stat_pooling,
        projector=rawnet3_projector,
        loss=loss,
    )
    if training:
        spk_model.train()
    else:
        spk_model.eval()

    kwargs = {"speech": inputs, "speech_lengths": ilens, "spk_labels": spk_labels}

    if training:
        loss, stats, weight = spk_model(**kwargs)
    else:
        loss, stats, weight = spk_model(**kwargs)
