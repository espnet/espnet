import pytest
import torch

from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.layers.create_adapter import create_adapter
from espnet2.layers.houlsby_adapter_layer import HoulsbyTransformerSentenceEncoderLayer

pytest.importorskip("transformers")
pytest.importorskip("s3prl")
pytest.importorskip("loralib")


def init_S3prl_model():
    class Model(torch.nn.Module):

        def __init__(self, frontend_conf: dict = {"upstream": "hubert_base"}):
            super().__init__()
            try:
                self.frontend = S3prlFrontend(frontend_conf=frontend_conf)
            except AttributeError:
                return None

    return Model()


def init_decoder_model():
    return TransformerDecoder(
        vocab_size=10,
        encoder_output_size=40,
        attention_heads=4,
        linear_units=40,
        num_blocks=2,
        input_layer="embed",
    )


@pytest.mark.parametrize(
    "model, adapter, adapter_conf",
    [(init_S3prl_model(), "houlsby", {"bottleneck": 64, "target_layers": []})],
)
def test_create_adapter_houslby(
    model,
    adapter,
    adapter_conf,
):
    if model is None:
        pytest.skip("Incorrect Initialization, probably due to torchaudio version.")
    create_adapter(model=model, adapter=adapter, adapter_conf=adapter_conf)
    assert isinstance(
        model.frontend.upstream.upstream.model.encoder.layers[0],
        HoulsbyTransformerSentenceEncoderLayer,
    )


@pytest.mark.parametrize(
    "model, adapter, adapter_conf",
    [
        (
            init_decoder_model(),
            "lora",
            {"rank": 2, "alpha": 4, "target_modules": ["linear_q"]},
        )
    ],
)
def test_create_adapter_lora(
    model,
    adapter,
    adapter_conf,
):
    create_adapter(model=model, adapter=adapter, adapter_conf=adapter_conf)
    assert hasattr(model.decoders[0].self_attn.linear_q, "lora_A")
    assert hasattr(model.decoders[0].self_attn.linear_q, "lora_B")


if __name__ == "__main__":
    test_create_adapter_houslby(
        init_S3prl_model(), "houlsby", {"bottleneck": 64, "target_layers": []}
    )
    print("Houlsby test passed")
    test_create_adapter_lora(
        init_decoder_model(),
        "lora",
        {"rank": 2, "alpha": 4, "target_modules": ["linear_q"]},
    )
    print("Lora test passed")
