import pytest
import torch

from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.ser.espnet_model import ESPnetSERModel
from espnet2.ser.loss.cross_entropy_loss import Xnt
from espnet2.ser.pooling.mean_pooling import MeanPooling
from espnet2.ser.projector.linear_projector import LinearProjector

frontend = S3prlFrontend(
    frontend_conf={"upstream": "wavlm_base"},
)

preencoder = LinearProjection(
    input_size=frontend.output_size(),
    output_size=16,
)

pooling = MeanPooling(input_size=preencoder.output_size())

linear_projector = LinearProjector(
    input_size=pooling.output_size(), output_size=4
)

xent_loss = Xnt(
    nout=linear_projector.output_size(),
    nclasses=linear_projector.output_size(),
)


@pytest.mark.parametrize("frontend", [frontend])
@pytest.mark.parametrize("preencoder", [preencoder])
@pytest.mark.parametrize("pooling", [pooling])
@pytest.mark.parametrize("projector", [linear_projector])
@pytest.mark.parametrize("loss", [xent_loss])
@pytest.mark.parametrize("training", [True, False])
def test_ser_model(frontend, preencoder, pooling, projector, loss, training):
    inputs = torch.randn(2, 10000)
    ilens = torch.LongTensor([8000, 10000])
    emotion_labels = torch.randint(0, 8, (2,1))
    ser_model = ESPnetSERModel(
        frontend=frontend,
        specaug=None,
        preencoder=preencoder,
        pooling=pooling,
        projector=projector,
        loss=loss,
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


@pytest.mark.parametrize("loss", [xent_loss])
@pytest.mark.parametrize("training", [True, False])
def test_ser_loss(training, loss):
    inputs = torch.randn(2, 10000)
    ilens = torch.LongTensor([8000, 10000])
    emotion_labels = torch.randint(0, 8, (2,1))
    ser_model = ESPnetSERModel(
        frontend=frontend,
        specaug=None,
        preencoder=preencoder,
        pooling=pooling,
        projector=linear_projector,
        loss=loss,
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

    if training:
        loss, stats, weight = ser_model(**kwargs)
    else:
        loss, stats, weight = ser_model(**kwargs)
