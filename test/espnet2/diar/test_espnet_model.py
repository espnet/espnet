import pytest
import torch

from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.diar.attractor.rnn_attractor import RnnAttractor
from espnet2.diar.decoder.linear_decoder import LinearDecoder
from espnet2.diar.espnet_model import ESPnetDiarizationModel
from espnet2.layers.label_aggregation import LabelAggregate

frontend = DefaultFrontend(
    n_fft=32,
    win_length=32,
    hop_length=16,
    n_mels=10,
)

encoder = TransformerEncoder(
    input_size=10,
    input_layer="linear",
    num_blocks=1,
    linear_units=32,
    output_size=16,
    attention_heads=2,
)

decoder = LinearDecoder(
    num_spk=2,
    encoder_output_size=encoder.output_size(),
)

rnn_attractor = RnnAttractor(unit=16, encoder_output_size=encoder.output_size())

label_aggregator = LabelAggregate(
    win_length=32,
    hop_length=16,
)


@pytest.mark.parametrize(
    "frontend, encoder, decoder, label_aggregator",
    [(frontend, encoder, decoder, label_aggregator)],
)
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("attractor", [rnn_attractor, None])
def test_single_channel_model(
    label_aggregator,
    frontend,
    encoder,
    decoder,
    attractor,
    training,
):
    inputs = torch.randn(2, 300)
    ilens = torch.LongTensor([300, 200])
    spk_labels = torch.randint(high=2, size=(2, 300, 2))
    diar_model = ESPnetDiarizationModel(
        label_aggregator=label_aggregator,
        attractor=attractor,
        encoder=encoder,
        decoder=decoder,
        frontend=frontend,
        specaug=None,
        normalize=None,
    )

    if training:
        diar_model.train()
    else:
        diar_model.eval()

    kwargs = {
        "speech": inputs,
        "speech_lengths": ilens,
        "spk_labels": spk_labels,
        "spk_labels_lengths": ilens,
    }
    loss, stats, weight = diar_model(**kwargs)
