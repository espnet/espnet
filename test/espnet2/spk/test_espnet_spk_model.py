import pytest
import torch

from espnet2.asr.frontend.asteroid_frontend import AsteroidFrontend
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.spk.encoder.rawnet3_encoder import RawNet3Encoder
from espnet2.spk.espnet_model import ESPnetSpeakerModel
from espnet2.spk.loss.aamsoftmax import AAMSoftmax
from espnet2.spk.pooling.chn_attn_stat_pooling import ChnAttnStatPooling
from espnet2.spk.projector.rawnet3_projector import RawNet3Projector

frontend = AsteroidFrontend(sinc_filters=256, sinc_stride=16)

rawnet3_encoder = RawNet3Encoder(
    input_size=frontend.output_size(), ndim=16, output_size=24
)

chn_attn_stat_pooling = ChnAttnStatPooling(input_size=rawnet3_encoder.output_size())

rawnet3_projector = RawNet3Projector(input_size=chn_attn_stat_pooling.output_size(), output_size=8)

aamsoftmax_loss = AAMSoftmax(nout=rawnet3_projector.output_size(), nclasses=10, margin=0.3, scale=15, easy_margin=False)
aamsoftmax_em_loss = AAMSoftmax(nout=rawnet3_projector.output_size(), nclasses=10, margin=0.3, scale=15, easy_margin=True)

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


@pytest.mark.parametrize("frontend", [frontend])
@pytest.mark.parametrize("encoder, projector", [(rawnet3_encoder, rawnet3_projector)])
@pytest.mark.parametrize("pooling", [chn_attn_stat_pooling])
@pytest.mark.parametrize("training", [True, False])
def test_single_channel_spk_model(frontend, encoder, pooling, projector, training):
    inputs = torch.randn(2, 8000)
    ilens = torch.LongTensor([8000, 7800])
    spk_labels = torch.randint(0, 10, (2,))
    spk_model = ESPnetSpeakerModel(
        frontend=frontend,
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
    spk_labels = torch.randint(0, 10, (2,))
    spk_model = ESPnetSpeakerModel(
        frontend=frontend,
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
