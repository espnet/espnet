import pytest

from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.layers.create_lora_adapter import create_lora_adapter

pytest.importorskip("lora")


def init_encoder():
    return TransformerEncoder(
        20,
        output_size=40,
        attention_heads=4,
        linear_units=40,
        num_blocks=2,
        input_layer="conv2d",
    )


def init_decoder():
    return TransformerDecoder(
        vocab_size=10,
        encoder_output_size=40,
        attention_heads=4,
        linear_units=40,
        num_blocks=2,
        input_layer="embed",
    )


@pytest.mark.parametrize(
    "rank, alpha, target_modules",
    [
        (2, 4, ["linear_q"]),
        (2, 4, ["linear_q", "linear_k", "linear_v", "linear_out"]),
    ],
)
def test_create_lora_adapter_encoder(rank, alpha, target_modules):
    model = init_encoder()
    create_lora_adapter(
        model=model, rank=rank, alpha=alpha, target_modules=target_modules
    )
    print(model)


@pytest.mark.parametrize(
    "rank, alpha, target_modules",
    [
        (2, 4, ["linear_q"]),
        (2, 4, ["linear_q", "linear_k", "linear_v", "linear_out"]),
        (2, 4, ["embed.0"]),  # Embedding layer
    ],
)
def test_create_lora_adapter_decoder(rank, alpha, target_modules):
    model = init_decoder()
    create_lora_adapter(
        model=model, rank=rank, alpha=alpha, target_modules=target_modules
    )
    print(model)


@pytest.mark.parametrize(
    "rank, alpha, target_modules",
    [(2, 4, ["linear"])],
)
def test_create_lora_adapter_invalid_target(rank, alpha, target_modules):
    model = init_encoder()
    with pytest.raises(ValueError):
        create_lora_adapter(
            model=model, rank=rank, alpha=alpha, target_modules=target_modules
        )
