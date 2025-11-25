import pytest
import torch
from packaging.version import parse as V

from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.ser.espnet_model import ESPnetSERModel
from espnet2.ser.loss.cross_entropy_loss import Xnt
from espnet2.ser.pooling.mean_pooling import MeanPooling
from espnet2.ser.projector.linear_projector import LinearProjector

is_torch_2_9_plus = V(torch.__version__) >= V("2.9.0")


@pytest.fixture
def setup_ser_components():
    """Setup SER model components for testing."""
    if is_torch_2_9_plus:
        pytest.skip("S3PRL is using unsupported attribute `set_audio_backend`.")

    frontend = S3prlFrontend(
        frontend_conf={"upstream": "wavlm_base"},
    )

    preencoder = LinearProjection(
        input_size=frontend.output_size(),
        output_size=16,
    )

    pooling = MeanPooling(input_size=preencoder.output_size())

    linear_projector = LinearProjector(input_size=pooling.output_size(), output_size=4)

    xent_loss = Xnt(
        nout=linear_projector.output_size(),
        nclasses=linear_projector.output_size(),
    )

    return {
        "frontend": frontend,
        "preencoder": preencoder,
        "pooling": pooling,
        "linear_projector": linear_projector,
        "xent_loss": xent_loss,
    }


@pytest.mark.parametrize("training", [True, False])
def test_ser_model(setup_ser_components, training):
    components = setup_ser_components
    inputs = torch.randn(2, 10000)
    ilens = torch.LongTensor([8000, 10000])
    emotion_labels = torch.randint(0, 4, (2, 1))
    ser_model = ESPnetSERModel(
        frontend=components["frontend"],
        specaug=None,
        preencoder=components["preencoder"],
        pooling=components["pooling"],
        projector=components["linear_projector"],
        loss=components["xent_loss"],
    )

    if training:
        ser_model.train()
    else:
        ser_model.eval()

    kwargs = {
        "speech": inputs,
        "speech_lengths": ilens,
        "emotion_labels": emotion_labels,
    }
    loss, stats, weight = ser_model(**kwargs)

    if training:
        loss.backward()


@pytest.mark.parametrize("training", [True, False])
def test_ser_loss(setup_ser_components, training):
    components = setup_ser_components
    inputs = torch.randn(2, 10000)
    ilens = torch.LongTensor([8000, 10000])
    emotion_labels = torch.randint(0, 4, (2, 1))
    ser_model = ESPnetSERModel(
        frontend=components["frontend"],
        specaug=None,
        preencoder=components["preencoder"],
        pooling=components["pooling"],
        projector=components["linear_projector"],
        loss=components["xent_loss"],
    )
    if training:
        ser_model.train()
    else:
        ser_model.eval()

    kwargs = {
        "speech": inputs,
        "speech_lengths": ilens,
        "emotion_labels": emotion_labels,
    }

    loss, stats, weight = ser_model(**kwargs)


def test_ser_prediction(setup_ser_components):
    """Test the prediction path (get_prediction=True)."""
    components = setup_ser_components
    inputs = torch.randn(2, 10000)
    ilens = torch.LongTensor([8000, 10000])
    ser_model = ESPnetSERModel(
        frontend=components["frontend"],
        specaug=None,
        preencoder=components["preencoder"],
        pooling=components["pooling"],
        projector=components["linear_projector"],
        loss=components["xent_loss"],
    )
    ser_model.eval()

    kwargs = {
        "speech": inputs,
        "speech_lengths": ilens,
        "get_prediction": True,
    }

    pred_emo = ser_model(**kwargs)
    assert pred_emo.shape == (2,)
    assert pred_emo.dtype == torch.long


def test_ser_collect_feats(setup_ser_components):
    """Test the collect_feats method."""
    components = setup_ser_components
    inputs = torch.randn(2, 10000)
    ilens = torch.LongTensor([8000, 10000])
    ser_model = ESPnetSERModel(
        frontend=components["frontend"],
        specaug=None,
        preencoder=components["preencoder"],
        pooling=components["pooling"],
        projector=components["linear_projector"],
        loss=components["xent_loss"],
    )

    feats_dict = ser_model.collect_feats(speech=inputs, speech_lengths=ilens)
    assert "feats" in feats_dict
    assert "feats_lengths" in feats_dict


def test_ser_extract_feats(setup_ser_components):
    """Test the extract_feats method."""
    components = setup_ser_components
    inputs = torch.randn(2, 10000)
    ilens = torch.LongTensor([8000, 10000])
    ser_model = ESPnetSERModel(
        frontend=components["frontend"],
        specaug=None,
        preencoder=components["preencoder"],
        pooling=components["pooling"],
        projector=components["linear_projector"],
        loss=components["xent_loss"],
    )

    feats, feats_lengths = ser_model.extract_feats(speech=inputs, speech_lengths=ilens)
    assert feats.shape[0] == 2


def test_ser_encode(setup_ser_components):
    """Test the encode method."""
    components = setup_ser_components
    inputs = torch.randn(2, 10000)
    ilens = torch.LongTensor([8000, 10000])
    ser_model = ESPnetSERModel(
        frontend=components["frontend"],
        specaug=None,
        preencoder=components["preencoder"],
        pooling=components["pooling"],
        projector=components["linear_projector"],
        loss=components["xent_loss"],
    )

    encoder_out, encoder_out_lens = ser_model.encode(speech=inputs, speech_lengths=ilens)
    assert encoder_out.shape[0] == 2
    assert encoder_out.shape[2] == components["preencoder"].output_size()


def test_ser_without_preencoder(setup_ser_components):
    """Test SER model without preencoder."""
    components = setup_ser_components
    inputs = torch.randn(2, 10000)
    ilens = torch.LongTensor([8000, 10000])
    emotion_labels = torch.randint(0, 4, (2, 1))

    pooling = MeanPooling(input_size=components["frontend"].output_size())
    linear_projector = LinearProjector(
        input_size=pooling.output_size(), output_size=4
    )
    xent_loss = Xnt(nout=linear_projector.output_size(), nclasses=4)

    ser_model = ESPnetSERModel(
        frontend=components["frontend"],
        specaug=None,
        preencoder=None,
        pooling=pooling,
        projector=linear_projector,
        loss=xent_loss,
    )

    kwargs = {
        "speech": inputs,
        "speech_lengths": ilens,
        "emotion_labels": emotion_labels,
    }
    loss, stats, weight = ser_model(**kwargs)
    assert loss is not None


def test_ser_without_frontend(setup_ser_components):
    """Test SER model without frontend."""
    components = setup_ser_components
    # Use feature input instead of raw audio
    inputs = torch.randn(2, 100, 80)  # (batch, time, feat_dim)
    ilens = torch.LongTensor([80, 100])
    emotion_labels = torch.randint(0, 4, (2, 1))

    pooling = MeanPooling(input_size=16)
    preencoder = LinearProjection(input_size=80, output_size=16)
    linear_projector = LinearProjector(input_size=pooling.output_size(), output_size=4)
    xent_loss = Xnt(nout=linear_projector.output_size(), nclasses=4)

    ser_model = ESPnetSERModel(
        frontend=None,
        specaug=None,
        preencoder=preencoder,
        pooling=pooling,
        projector=linear_projector,
        loss=xent_loss,
    )

    kwargs = {
        "speech": inputs,
        "speech_lengths": ilens,
        "emotion_labels": emotion_labels,
    }
    loss, stats, weight = ser_model(**kwargs)
    assert loss is not None
