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
