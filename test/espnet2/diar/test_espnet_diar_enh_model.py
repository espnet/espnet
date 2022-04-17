from distutils.version import LooseVersion

import pytest
import torch

from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.diar.attractor.rnn_attractor import RnnAttractor
from espnet2.diar.decoder.linear_decoder import LinearDecoder
from espnet2.diar.espnet_diar_enh_model import ESPnetDiarEnhModel
from espnet2.diar.layers.multi_mask import MultiMask
from espnet2.diar.separator.tcn_separator_nomask import TCNSeparator
from espnet2.enh.decoder.conv_decoder import ConvDecoder
from espnet2.enh.encoder.conv_encoder import ConvEncoder
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainL1
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainMSE
from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.enh.loss.wrappers.fixed_order import FixedOrderSolver
from espnet2.enh.loss.wrappers.pit_solver import PITSolver
from espnet2.layers.label_aggregation import LabelAggregate

is_torch_1_9_plus = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")

label_aggregator = LabelAggregate(
    win_length=32,
    hop_length=16,
)

enh_encoder = ConvEncoder(
    channel=17,
    kernel_size=32,
    stride=16,
)

enh_decoder = ConvDecoder(
    channel=17,
    kernel_size=32,
    stride=16,
)

tcn_separator = TCNSeparator(
    input_dim=enh_encoder.output_dim,
    layer=2,
    stack=1,
    bottleneck_dim=10,
    hidden_dim=10,
    kernel=3,
)

mask_module = MultiMask(
    bottleneck_dim=10,
    max_num_spk=3,
    input_dim=enh_encoder.output_dim,
)

diar_encoder = TransformerEncoder(
    input_layer="linear",
    num_blocks=1,
    linear_units=32,
    output_size=16,
    attention_heads=2,
    input_size=tcn_separator.output_dim,
)

diar_decoder = LinearDecoder(
    num_spk=2,
    encoder_output_size=diar_encoder.output_size(),
)

rnn_attractor = RnnAttractor(unit=16, encoder_output_size=diar_encoder.output_size())

si_snr_loss = SISNRLoss()
tf_mse_loss = FrequencyDomainMSE()
tf_l1_loss = FrequencyDomainL1()

pit_wrapper = PITSolver(criterion=si_snr_loss)
fix_order_solver = FixedOrderSolver(criterion=tf_mse_loss)


@pytest.mark.parametrize("label_aggregator", [label_aggregator])
@pytest.mark.parametrize("enh_encoder, enh_decoder", [(enh_encoder, enh_decoder)])
@pytest.mark.parametrize("separator", [tcn_separator])
@pytest.mark.parametrize("mask_module", [mask_module])
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("loss_wrappers", [[pit_wrapper, fix_order_solver]])
@pytest.mark.parametrize("diar_encoder, diar_decoder", [(diar_encoder, diar_decoder)])
@pytest.mark.parametrize("attractor", [rnn_attractor, None])
def test_single_channel_model(
    label_aggregator,
    diar_encoder,
    diar_decoder,
    attractor,
    enh_encoder,
    enh_decoder,
    separator,
    mask_module,
    training,
    loss_wrappers,
):
    inputs = torch.randn(2, 300)
    ilens = torch.LongTensor([300, 200])
    speech_refs = [torch.randn(2, 300).float(), torch.randn(2, 300).float()]
    spk_labels = torch.randint(high=2, size=(2, 300, 2))
    enh_model = ESPnetDiarEnhModel(
        label_aggregator=label_aggregator,
        attractor=attractor,
        enh_encoder=enh_encoder,
        separator=separator,
        mask_module=mask_module,
        enh_decoder=enh_decoder,
        diar_encoder=diar_encoder,
        diar_decoder=diar_decoder,
        loss_wrappers=loss_wrappers,
        frontend=None,
        specaug=None,
        normalize=None,
    )

    if training:
        enh_model.train()
    else:
        enh_model.eval()

    kwargs = {
        "speech_mix": inputs,
        "speech_mix_lengths": ilens,
        "spk_labels": spk_labels,
        "spk_labels_lengths": ilens,
        **{"speech_ref{}".format(i + 1): speech_refs[i] for i in range(2)},
    }
    loss, stats, weight = enh_model(**kwargs)
