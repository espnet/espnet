import pytest
import torch
from packaging.version import parse as V

from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.frontend.cnn import CNNFrontend
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.ssl.espnet_model import ESPnetSSLModel
from espnet2.ssl.loss.hubert import HuBERTLoss
from espnet2.tasks.ssl import util_choices
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


@pytest.mark.parametrize("encoder_arch", [TransformerEncoder])
@pytest.mark.parametrize("loss_fn", [HuBERTLoss])
def test_espnet_model_wav2vec(encoder_arch, loss_fn):
    frontend = CNNFrontend("group_norm", "standard", True, [(3, 3, 2)])
    preencoder = LinearProjection(3, 16)
    encoder = encoder_arch(
        16,
        output_size=16,
        attention_heads=1,
        linear_units=4,
        num_blocks=2,
        pos_enc_layer_type="conv",
        input_layer="wav2vec",
    )
    losses = [loss_fn(16, 5, 1)]
    util_attributes = set()
    required_inputs = set()

    for loss_func in losses:
        util_attributes.update(loss_func.util_attributes)
        required_inputs.update(loss_func.required_inputs)
    util_modules = torch.nn.ModuleDict()
    for attr in util_attributes:
        util_args = {}
        util_class = util_choices.get_class(attr)
        if attr == "ema":
            util_args["model"] = encoder
            util_args["device"] = f"cuda:{torch.cuda.current_device()}"
        if attr == "mask":
            util_args["encoder_embed_dim"] = encoder.output_size()
        util = util_class(**util_args)
        util_modules.update({attr: util})

    model = ESPnetSSLModel(
        frontend=frontend,
        specaug=None,
        normalize=None,
        preencoder=preencoder,
        encoder=encoder,
        losses=losses,
        util_attributes=util_attributes,
        required_inputs=required_inputs,
        util_modules=util_modules,
        token_list=["0", "1", "2", "3"],
    )

    inputs = dict(
        speech=torch.randn(2, 32, requires_grad=True),
        speech_lengths=torch.tensor([32, 16], dtype=torch.long),
        text=torch.randint(0, 5, [2, 15], dtype=torch.long),
        text_lengths=torch.tensor([15, 7], dtype=torch.long),
    )
    loss, *_ = model(**inputs)
    loss.backward()


@pytest.mark.parametrize("encoder_arch", [EBranchformerEncoder, ConformerEncoder])
@pytest.mark.parametrize("loss_fn", [HuBERTLoss])
def test_espnet_model_conformer(encoder_arch, loss_fn):
    frontend = CNNFrontend("group_norm", "standard", True, [(3, 3, 2)])
    preencoder = LinearProjection(3, 16)
    encoder = encoder_arch(
        16,
        output_size=16,
        attention_heads=1,
        linear_units=4,
        num_blocks=2,
        input_layer=None,
    )
    losses = [loss_fn(16, 5, 1)]
    util_attributes = set()
    required_inputs = set()

    for loss_func in losses:
        util_attributes.update(loss_func.util_attributes)
        required_inputs.update(loss_func.required_inputs)
    util_modules = torch.nn.ModuleDict()
    for attr in util_attributes:
        util_args = {}
        util_class = util_choices.get_class(attr)
        if attr == "ema":
            util_args["model"] = encoder
            util_args["device"] = f"cuda:{torch.cuda.current_device()}"
        if attr == "mask":
            util_args["encoder_embed_dim"] = encoder.output_size()
        util = util_class(**util_args)
        util_modules.update({attr: util})

    model = ESPnetSSLModel(
        frontend=frontend,
        specaug=None,
        normalize=None,
        preencoder=preencoder,
        encoder=encoder,
        losses=losses,
        util_attributes=util_attributes,
        required_inputs=required_inputs,
        util_modules=util_modules,
        token_list=["0", "1", "2", "3"],
    )

    inputs = dict(
        speech=torch.randn(2, 32, requires_grad=True),
        speech_lengths=torch.tensor([32, 16], dtype=torch.long),
        text=torch.randint(0, 5, [2, 15], dtype=torch.long),
        text_lengths=torch.tensor([15, 7], dtype=torch.long),
    )
    loss, *_ = model(**inputs)
    loss.backward()


def test_masking():
    util_class = util_choices.get_class("mask")
    masking = util_class(encoder_embed_dim=16)

    data = dict(
        encoder_output=torch.randn(2, 16, 16, requires_grad=False),
        encoder_output_lengths=torch.tensor([16, 8], dtype=torch.long),
    )

    # check if mask was applied correctly
    encoded = data["encoder_output"]
    encoded_lengths = data["encoder_output_lengths"]
    pad_masks = make_pad_mask(encoded_lengths).to(encoded.device)

    masked, mask_info = masking(encoded, pad_masks)
    mask_m = mask_info["mask_m"]
    mask_u = mask_info["mask_u"]

    masked_sample = masked[mask_m][0]
    unmasked_sample = masked[mask_u][0]
    assert masked_sample.eq(masking.mask_embedding.data).all()
    assert unmasked_sample.ne(masking.mask_embedding.data).all()

    # check mask shape
    # masking changes B,T,D to BT, D
    assert encoded[mask_m].size(0) <= encoded.size(0) * encoded.size(1)
    assert encoded[mask_u].size(0) <= encoded.size(0) * encoded.size(1)
