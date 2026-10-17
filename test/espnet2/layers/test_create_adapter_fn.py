import pytest
import torch
from typeguard import TypeCheckError

from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.layers.create_adapter_fn import create_houlsby_adapter, create_lora_adapter
from espnet2.layers.houlsby_adapter_layer import (  # Houlsby_Adapter,
    HoulsbyTransformerSentenceEncoderLayer,
)

pytest.importorskip("transformers")
pytest.importorskip("s3prl")
pytest.importorskip("loralib")


def init_S3prl_model(frontend_conf={"upstream": "hubert_base"}):
    class Model(torch.nn.Module):

        def __init__(self, frontend_conf: dict = {"upstream": "hubert_base"}):
            super().__init__()
            print(frontend_conf)
            self.frontend = S3prlFrontend(frontend_conf=frontend_conf)

    return Model(frontend_conf)


def init_decoder_model():
    return TransformerDecoder(
        vocab_size=10,
        encoder_output_size=40,
        attention_heads=4,
        linear_units=40,
        num_blocks=2,
        input_layer="embed",
    )


# =========================================Houlsby================================================
# These four tests build an s3prl frontend, and the first one to run downloads
# the upstream it asks for - hubert_base is a few hundred MB. The 20 s budget
# they carried predates that being on the clock: the tests spent six months
# skipped behind a torch version check, and were re-enabled in #6678. Two
# consecutive CI runs then failed here, one on a connection reset mid-download
# and one on the 20 s timeout itself, on a change that touches none of this.
# 120 s covers a cold cache on a slow morning. If it still flakes, the answer is
# to fetch the upstream when the CI image is built rather than to keep raising
# this number.
@pytest.mark.execution_timeout(120)
@pytest.mark.parametrize("model, bottleneck, target_layers", [("s3prl", 64, [])])
def test_create_houlsby_adapter_bottleneck(
    model,
    bottleneck,
    target_layers,
):
    assert model == "s3prl"
    model = init_S3prl_model()
    create_houlsby_adapter(
        model=model, bottleneck=bottleneck, target_layers=target_layers
    )
    assert (
        model.frontend.upstream.upstream.model.encoder.layers[0].bottleneck
        == bottleneck
    )


@pytest.mark.execution_timeout(120)
@pytest.mark.parametrize(
    "model, bottleneck, target_layers",
    [
        (
            "s3prl",
            64,
            [],
        )
    ],
)
def test_create_houlsby_adapter_hf_wav2vec2_custom_bottleneck(
    model,
    bottleneck,
    target_layers,
):
    assert model == "s3prl"
    model = init_S3prl_model(
        frontend_conf={
            "upstream": "hf_wav2vec2_custom",
            "path_or_url": "facebook/mms-300m",
        }
    )
    create_houlsby_adapter(
        model=model, bottleneck=bottleneck, target_layers=target_layers
    )
    assert (
        model.frontend.upstream.upstream.model.encoder.layers[0].bottleneck
        == bottleneck
    )


@pytest.mark.execution_timeout(120)
@pytest.mark.parametrize("model, bottleneck, target_layers", [("s3prl", 64, [1, 2])])
def test_create_houlsby_adapter_target_layers(
    model,
    bottleneck,
    target_layers,
):
    assert model == "s3prl"
    model = init_S3prl_model()
    create_houlsby_adapter(
        model=model, bottleneck=bottleneck, target_layers=target_layers
    )
    assert not isinstance(
        model.frontend.upstream.upstream.model.encoder.layers[0],
        HoulsbyTransformerSentenceEncoderLayer,
    ), type(model.frontend.upstream.upstream.model.encoder.layers[0])
    assert isinstance(
        model.frontend.upstream.upstream.model.encoder.layers[1],
        HoulsbyTransformerSentenceEncoderLayer,
    ), type(model.frontend.upstream.upstream.model.encoder.layers[1])
    assert isinstance(
        model.frontend.upstream.upstream.model.encoder.layers[2],
        HoulsbyTransformerSentenceEncoderLayer,
    ), type(model.frontend.upstream.upstream.model.encoder.layers[2])
    assert not isinstance(
        model.frontend.upstream.upstream.model.encoder.layers[3],
        HoulsbyTransformerSentenceEncoderLayer,
    ), type(model.frontend.upstream.upstream.model.encoder.layers[3])


@pytest.mark.execution_timeout(120)
@pytest.mark.parametrize("model, bottleneck, target_layers", [("s3prl", 64, [200])])
def test_create_houlsby_adapter_invalid_target_layers(
    model,
    bottleneck,
    target_layers,
):
    assert model == "s3prl"
    model = init_S3prl_model()
    with pytest.raises(ValueError):
        create_houlsby_adapter(
            model=model, bottleneck=bottleneck, target_layers=target_layers
        )


@pytest.mark.parametrize(
    "model, bottleneck, target_layers", [(init_decoder_model(), 64, [])]
)
def test_create_houlsby_adapter_invalid_model(
    model,
    bottleneck,
    target_layers,
):
    with pytest.raises(AssertionError):
        create_houlsby_adapter(
            model=model, bottleneck=bottleneck, target_layers=target_layers
        )


# =========================================LORA================================================
@pytest.mark.parametrize("rank, alpha, target_modules", [(2, 4, ["linear_q"])])
def test_create_lora_adapter_linear(rank, alpha, target_modules):
    model = init_decoder_model()
    create_lora_adapter(
        model=model, rank=rank, alpha=alpha, target_modules=target_modules
    )

    assert model.decoders[0].self_attn.linear_q.lora_A.shape[0] == rank
    assert model.decoders[0].self_attn.linear_q.lora_B.shape[1] == rank


@pytest.mark.parametrize("rank, alpha, target_modules", [(2, 4, ["embed.0"])])
def test_create_lora_adapter_embedding(rank, alpha, target_modules):
    model = init_decoder_model()
    create_lora_adapter(
        model=model, rank=rank, alpha=alpha, target_modules=target_modules
    )

    assert model.embed[0].lora_A.shape[0] == rank
    assert model.embed[0].lora_B.shape[1] == rank


@pytest.mark.parametrize("rank, alpha, target_modules", [(2, 4, ["query_proj"])])
def test_create_lora_adapter_invalid_target(rank, alpha, target_modules):
    model = init_decoder_model()
    with pytest.raises(ValueError):
        create_lora_adapter(
            model=model, rank=rank, alpha=alpha, target_modules=target_modules
        )


@pytest.mark.parametrize("rank, alpha, target_modules", [(2, 4, ["norm1"])])
def test_create_lora_adapter_unsupport_target(rank, alpha, target_modules):
    model = init_decoder_model()
    with pytest.raises(ValueError):
        create_lora_adapter(
            model=model, rank=rank, alpha=alpha, target_modules=target_modules
        )


@pytest.mark.parametrize("rank, alpha, target_modules", [(2, 4, 5)])
def test_create_lora_adapter_invalid_type(rank, alpha, target_modules):
    model = init_decoder_model()
    with pytest.raises(TypeCheckError):
        create_lora_adapter(
            model=model, rank=rank, alpha=alpha, target_modules=target_modules
        )


if __name__ == "__main__":
    s3prl_model = init_S3prl_model()
    test_create_houlsby_adapter_bottleneck("s3prl", 64, [])
    print("create_houlsby_adapter_bottleneck test passed")
    print("-----------------------------------------------------------")
    s3prl_model = init_S3prl_model(
        {"upstream": "hf_wav2vec2_custom", "path_or_url": "facebook/mms-300m"}
    )
    test_create_houlsby_adapter_hf_wav2vec2_custom_bottleneck("s3prl", 64, [])
    print("create_houlsby_adapter_hf_wav2vec2_custom_bottleneck test passed")
    print("-----------------------------------------------------------")
    exit()
    s3prl_model = init_S3prl_model()
    test_create_houlsby_adapter_target_layers(s3prl_model, 64, [1, 2])
    print("create_houlsby_adapter_target_layers test passed")
    print("-----------------------------------------------------------")
    s3prl_model = init_S3prl_model()
    test_create_houlsby_adapter_invalid_target_layers(s3prl_model, 64, [200])
    print("create_houlsby_adapter_invalid_target_layers test passed")
    print("-----------------------------------------------------------")
    decoder_model = init_decoder_model()
    test_create_houlsby_adapter_invalid_model(decoder_model, 64, [2])
    print("create_houlsby_adapter_invalid_model test passed")
    print("-----------------------------------------------------------")
    decoder_model = init_decoder_model()
    test_create_lora_adapter_embedding(2, 4, ["embed.0"])
    print("create_lora_adapter_embedding test passed")
    print("-----------------------------------------------------------")
    decoder_model = init_decoder_model()
    test_create_lora_adapter_invalid_target(2, 4, ["query_proj"])
    print("create_lora_adapter_invalid_target test passed")
    print("-----------------------------------------------------------")
    decoder_model = init_decoder_model()
    test_create_lora_adapter_unsupport_target(2, 4, ["norm1"])
    print("create_lora_adapter_unsupport_target test passed")
    print("-----------------------------------------------------------")
    decoder_model = init_decoder_model()
    test_create_lora_adapter_invalid_type(2, 4, 5)
    print("create_lora_adapter_invalid_type test passed")
    print("-----------------------------------------------------------")

    print("create_adapter_fn test passed")
