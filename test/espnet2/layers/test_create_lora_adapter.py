import sys

import pytest
import torch
from packaging.version import parse as V

from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.layers.create_lora_adapter import create_lora_adapter

is_python_3_8_plus = sys.version_info >= (3, 8)
is_torch_1_8_plus = V(torch.__version__) >= V("1.8.0")


def init_model():
    return TransformerDecoder(
        vocab_size=10,
        encoder_output_size=40,
        attention_heads=4,
        linear_units=40,
        num_blocks=2,
        input_layer="embed",
    )


@pytest.mark.skipif(
    not is_torch_1_8_plus or not is_python_3_8_plus, reason="Not supported"
)
@pytest.mark.parametrize("rank, alpha, target_modules", [(2, 4, ["linear_q"])])
def test_create_lora_adapter_linear(rank, alpha, target_modules):
    model = init_model()
    create_lora_adapter(
        model=model, rank=rank, alpha=alpha, target_modules=target_modules
    )

    assert model.decoders[0].self_attn.linear_q.lora_A.shape[0] == rank
    assert model.decoders[0].self_attn.linear_q.lora_B.shape[1] == rank


@pytest.mark.skipif(
    not is_torch_1_8_plus or not is_python_3_8_plus, reason="Not supported"
)
@pytest.mark.parametrize("rank, alpha, target_modules", [(2, 4, ["embed.0"])])
def test_create_lora_adapter_embedding(rank, alpha, target_modules):
    model = init_model()
    create_lora_adapter(
        model=model, rank=rank, alpha=alpha, target_modules=target_modules
    )

    assert model.embed[0].lora_A.shape[0] == rank
    assert model.embed[0].lora_B.shape[1] == rank


@pytest.mark.skipif(
    not is_torch_1_8_plus or not is_python_3_8_plus, reason="Not supported"
)
@pytest.mark.parametrize("rank, alpha, target_modules", [(2, 4, ["query_proj"])])
def test_create_lora_adapter_invalid_target(rank, alpha, target_modules):
    model = init_model()
    with pytest.raises(ValueError):
        create_lora_adapter(
            model=model, rank=rank, alpha=alpha, target_modules=target_modules
        )


@pytest.mark.skipif(
    not is_torch_1_8_plus or not is_python_3_8_plus, reason="Not supported"
)
@pytest.mark.parametrize("rank, alpha, target_modules", [(2, 4, ["norm1"])])
def test_create_lora_adapter_unsupport_target(rank, alpha, target_modules):
    model = init_model()
    with pytest.raises(ValueError):
        create_lora_adapter(
            model=model, rank=rank, alpha=alpha, target_modules=target_modules
        )


@pytest.mark.skipif(
    not is_torch_1_8_plus or not is_python_3_8_plus, reason="Not supported"
)
@pytest.mark.parametrize("rank, alpha, target_modules", [(2, 4, 5)])
def test_create_lora_adapter_invalid_type(rank, alpha, target_modules):
    model = init_model()
    with pytest.raises(TypeError):
        create_lora_adapter(
            model=model, rank=rank, alpha=alpha, target_modules=target_modules
        )
