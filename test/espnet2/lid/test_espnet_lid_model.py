import pytest
import torch

from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.lid.espnet_model import ESPnetLIDModel
from espnet2.spk.encoder.ecapa_tdnn_encoder import EcapaTdnnEncoder
from espnet2.spk.loss.aamsoftmax_subcenter_intertopk import (
    ArcMarginProduct_intertopk_subcenter,
)
from espnet2.spk.pooling.chn_attn_stat_pooling import ChnAttnStatPooling
from espnet2.spk.projector.rawnet3_projector import RawNet3Projector

default_frontend = DefaultFrontend(
    fs=16000,
    n_mels=80,
)

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

ecapa_tdnn_encoder = EcapaTdnnEncoder(
    default_frontend.output_size(), model_scale=2, ndim=16, output_size=24
)

chan_attn_stat_pooling = ChnAttnStatPooling(input_size=ecapa_tdnn_encoder.output_size())

rawnet3_projector = RawNet3Projector(
    input_size=chan_attn_stat_pooling.output_size(), output_size=8
)

aamsoftmax_it_sub_loss = ArcMarginProduct_intertopk_subcenter(
    nout=rawnet3_projector.output_size(),
    nclasses=10,
    margin=0.3,
    scale=15,
    K=2,
    mp=0.06,
    k_top=2,
)


@pytest.mark.parametrize("frontend", [default_frontend])
@pytest.mark.parametrize("specaug", [specaug])
@pytest.mark.parametrize("normalize", [normalize])
@pytest.mark.parametrize("encoder", [ecapa_tdnn_encoder])
@pytest.mark.parametrize("pooling", [chan_attn_stat_pooling])
@pytest.mark.parametrize("projector", [rawnet3_projector])
@pytest.mark.parametrize("loss", [aamsoftmax_it_sub_loss])
@pytest.mark.parametrize("training", [True, False])
def test_lid_model(
    frontend, specaug, normalize, encoder, pooling, projector, loss, training
):
    inputs = torch.randn(2, 8000)
    ilens = torch.LongTensor([8000, 7800])
    lid_labels = torch.randint(0, 10, (2,))
    lid_model = ESPnetLIDModel(
        frontend=frontend,
        specaug=specaug,
        normalize=normalize,
        encoder=encoder,
        pooling=pooling,
        projector=projector,
        loss=loss,
    )

    if training:
        lid_model.train()
    else:
        lid_model.eval()

    kwargs = {"speech": inputs, "speech_lengths": ilens, "lid_labels": lid_labels}
    loss, *_ = lid_model(**kwargs)
    loss.backward()
