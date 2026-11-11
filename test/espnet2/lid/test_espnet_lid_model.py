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

    if training:
        loss, *_ = lid_model(**kwargs)
        loss.backward()
    else:
        loss, *_ = lid_model(**kwargs)


@pytest.fixture(params=[None, "default"])
def plain_lid_args(request):
    from espnet2.tasks.lid import LIDTask

    args = LIDTask.get_parser().parse_args([])
    args.model_conf = {}
    args.frontend = request.param
    args.frontend_conf = dict(n_fft=64, hop_length=16, n_mels=8)
    args.input_size = 8
    args.encoder = "identity"
    args.pooling = "mean"
    args.projector = "rawnet3"
    args.projector_conf = dict(output_size=4)
    args.loss = "softmax"
    args.lang_num = 4
    return args


def test_default_lid_checkpoint_and_forward_contract(plain_lid_args):
    from espnet2.tasks.lid import LIDTask

    model = LIDTask.build_model(plain_lid_args).eval()
    # Existing LID checkpoints contain only the classification architecture.
    expected_keys = {
        "projector.bn.weight",
        "projector.bn.bias",
        "projector.bn.running_mean",
        "projector.bn.running_var",
        "projector.bn.num_batches_tracked",
        "projector.fc.weight",
        "projector.fc.bias",
        "loss.weight",
    }
    if model.frontend is not None:
        expected_keys.add("frontend.logmel.melmat")
    assert set(model.state_dict()) == expected_keys
    assert model.lang2vec_head is None
    assert model.conditioned_layers == []
    lengths = torch.tensor([320, 280])
    speech = (
        torch.randn(2, 320) if model.frontend is not None else torch.randn(2, 320, 8)
    )
    labels = torch.tensor([0, 1])
    feats, feat_lengths = model.extract_feats(speech, lengths)
    assert model.collect_feats(speech, lengths)["feats"].shape == feats.shape
    expected_embeddings = model.projector(
        model.pooling(feats.transpose(1, 2), feat_lengths=feat_lengths)
    )
    expected_logits = torch.nn.functional.linear(
        torch.nn.functional.normalize(expected_embeddings),
        torch.nn.functional.normalize(model.loss.weight),
    )
    expected_loss = torch.nn.functional.cross_entropy(expected_logits, labels)
    # Extra data fields must not activate auxiliary training on ordinary LID.
    loss, stats, weight = model(
        speech, lengths, labels, lang2vecs=torch.full((2, 3), float("nan"))
    )
    torch.testing.assert_close(loss, expected_loss.reshape_as(loss))
    assert set(stats) == {"loss", "accuracy"}
    assert weight == 2
    # Retain the original fourth positional argument for embedding extraction.
    embeddings, predictions = model(speech, lengths, None, True)
    torch.testing.assert_close(embeddings, expected_embeddings)
    torch.testing.assert_close(predictions, expected_logits.argmax(dim=-1))
    loss.backward()
    assert model.loss.weight.grad is not None
    assert model.projector.fc.weight.grad is not None


def test_downstream_auxiliary_with_non_ssl_frontend(plain_lid_args):
    from espnet2.tasks.lid import LIDTask

    plain_lid_args.model_conf = dict(lang2vec_dim=3, lang2vec_weight=0.2)
    model = LIDTask.build_model(plain_lid_args)
    assert model.conditioned_layers == []
    speech = (
        torch.randn(2, 320) if model.frontend is not None else torch.randn(2, 320, 8)
    )
    loss, stats, _ = model(
        speech,
        torch.tensor([320, 280]),
        lid_labels=torch.tensor([0, 1]),
        lang2vecs=torch.randn(2, 3),
    )
    torch.testing.assert_close(
        loss.detach(),
        0.8 * stats["class_loss"] + 0.2 * stats["lang2vec_loss_downstream"],
    )
    loss.backward()
    assert model.lang2vec_head.weight.grad.abs().sum() > 0
    assert model.loss.weight.grad.abs().sum() > 0


def test_conditioning_requires_ssl_frontend(plain_lid_args):
    from espnet2.tasks.lid import LIDTask

    plain_lid_args.model_conf = dict(lang2vec_conditioning_layers=[0])
    plain_lid_args.encoder_condition = "identity"
    plain_lid_args.pooling_condition = "mean"
    plain_lid_args.projector_condition = "rawnet3"
    with pytest.raises(ValueError, match="require S3prlFrontend"):
        LIDTask.build_model(plain_lid_args)


def _mfa_e_branchformer_model():
    from espnet2.lid.loss.linear_softmax import LinearSoftmax
    from espnet2.spk.encoder.e_branchformer_encoder import MfaEBranchformerEncoder
    from espnet2.spk.pooling.stat_pooling import StatsPooling

    torch.manual_seed(0)
    encoder = MfaEBranchformerEncoder(
        input_size=20,
        output_size=16,
        attention_heads=2,
        linear_units=32,
        cgmlp_linear_units=32,
        num_blocks=2,
        input_layer="conv2d",
        attention_layer_type="selfattn",
        pos_enc_layer_type="abs_pos",
        use_ffn=True,
        macaron_ffn=True,
        merge_conv_kernel=3,
        cgmlp_conv_kernel=3,
        use_flash_attn=False,
    )
    pooling = StatsPooling(input_size=encoder.output_size())
    projector = RawNet3Projector(input_size=pooling.output_size(), output_size=8)
    return ESPnetLIDModel(
        frontend=DefaultFrontend(fs=16000, n_mels=20),
        specaug=None,
        normalize=UtteranceMVN(norm_means=True, norm_vars=False),
        encoder=encoder,
        pooling=pooling,
        projector=projector,
        loss=LinearSoftmax(nout=projector.output_size(), nclasses=5),
    )


@pytest.mark.parametrize("training", [True, False])
def test_lid_model_with_mfa_e_branchformer(training):
    model = _mfa_e_branchformer_model().train(training)
    speech, lengths = torch.randn(2, 8000), torch.tensor([8000, 5600])
    loss, stats, _ = model(speech, lengths, lid_labels=torch.tensor([0, 3]))
    loss.backward()
    assert torch.isfinite(loss) and "accuracy" in stats
    model.eval()
    embeddings, predictions = model(speech, lengths, extract_embd=True)
    assert embeddings.shape == (2, 8)
    assert torch.equal(predictions, model.loss.linear(embeddings).argmax(1))


def test_mfa_e_branchformer_embedding_ignores_padding():
    # From the features on: the STFT frontend itself lets the last frames of
    # an utterance see the padding (centered windows), for every encoder.
    model = _mfa_e_branchformer_model().eval()

    def embed(feats, lengths):
        frames, frame_lengths = model.encode_frame_with_lengths(feats, lengths)
        utt = model.pooling(frames, feat_lengths=frame_lengths)
        return model.project_lang_embd(utt)

    feats = torch.randn(1, 37, 20)
    alone = embed(feats, torch.tensor([37]))
    padded = torch.cat([feats, torch.randn(1, 24, 20) * 5], dim=1)
    batch = embed(padded, torch.tensor([37]))
    torch.testing.assert_close(alone, batch, atol=1e-5, rtol=1e-5)


def test_mfa_e_branchformer_encode_frame_returns_the_frames():
    model = _mfa_e_branchformer_model().eval()
    feats = torch.randn(2, 37, 20)
    frames, _ = model.encode_frame_with_lengths(feats, torch.tensor([37, 37]))
    torch.testing.assert_close(model.encode_frame(feats), frames)


def test_tdnn_encoders_keep_the_input_lengths():
    """ECAPA-TDNN models compute the same embeddings as before."""
    model = ESPnetLIDModel(
        frontend=default_frontend,
        specaug=None,
        normalize=normalize,
        encoder=ecapa_tdnn_encoder,
        pooling=chan_attn_stat_pooling,
        projector=rawnet3_projector,
        loss=aamsoftmax_it_sub_loss,
    ).eval()
    feats, lengths = torch.randn(2, 30, 80), torch.tensor([30, 21])
    out, out_lengths = model.encode_frame_with_lengths(feats, lengths)
    assert out_lengths is lengths
    assert torch.equal(out, model.encode_frame(feats))

    speech, speech_lengths = torch.randn(2, 8000), torch.tensor([8000, 5600])
    embeddings, _ = model(speech, speech_lengths, extract_embd=True)
    feats, feat_lengths = model.extract_feats(speech, speech_lengths)
    expected = model.projector(
        model.pooling(model.encoder(feats), feat_lengths=feat_lengths)
    )
    torch.testing.assert_close(embeddings, expected)
