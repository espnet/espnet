import json
from pathlib import Path

import pytest
import torch

from espnet2.asr.decoder.hugging_face_transformers_decoder import (
    HuggingFaceTransformersDecoder,
)


@pytest.mark.parametrize("causal_lm", [False, True])
@pytest.mark.parametrize("as_json", [False, True])
def test_architecture_override_cannot_enable_custom_code(tmp_path, causal_lm, as_json):
    # A local model requiring custom code exercises Transformers' real loader
    # without downloading or running code from a third-party repository.
    model_dir = tmp_path / "custom_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "espnet_test_custom",
                "auto_map": {"AutoConfig": "custom_config.CustomConfig"},
            }
        )
    )
    (model_dir / "custom_config.py").write_text(
        'raise AssertionError("Custom configuration code was executed")\n'
    )
    overrides = {"trust_remote_code": True}
    if as_json:
        config_path = tmp_path / "overrides.json"
        config_path.write_text(json.dumps(overrides))
        overrides = str(config_path)

    original_equal = torch.equal
    with pytest.raises(ValueError, match="trust_remote_code"):
        HuggingFaceTransformersDecoder(
            vocab_size=16,
            encoder_output_size=8,
            model_name_or_path=str(model_dir),
            causal_lm=causal_lm,
            overriding_architecture_config=overrides,
        )
    assert torch.equal is original_equal


@pytest.fixture(params=[False, True], ids=["seq2seq", "causal"])
def local_hf_model(tmp_path, request):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (
        BloomConfig,
        BloomForCausalLM,
        MBartConfig,
        MBartForConditionalGeneration,
        PreTrainedTokenizerFast,
    )

    causal_lm = request.param
    if causal_lm:
        model = BloomForCausalLM(
            BloomConfig(vocab_size=16, hidden_size=8, n_layer=1, n_head=2)
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
            unk_token="[UNK]",
        )
        tokenizer.save_pretrained(tmp_path)
    else:
        model = MBartForConditionalGeneration(
            MBartConfig(
                vocab_size=16,
                d_model=8,
                encoder_layers=1,
                decoder_layers=1,
                encoder_attention_heads=2,
                decoder_attention_heads=2,
                encoder_ffn_dim=16,
                decoder_ffn_dim=16,
            )
        )
    model.save_pretrained(tmp_path)
    return str(tmp_path), causal_lm


@pytest.mark.parametrize("as_json", [False, True])
def test_safe_architecture_overrides_load_local_model(
    local_hf_model, tmp_path, as_json
):
    model_path, causal_lm = local_hf_model
    overrides = {"use_cache": False, "trust_remote_code": True}
    config = overrides
    if as_json:
        config_path = tmp_path / "overrides.json"
        config_path.write_text(json.dumps(overrides))
        config = str(config_path)

    decoder = HuggingFaceTransformersDecoder(
        vocab_size=16,
        encoder_output_size=8,
        model_name_or_path=model_path,
        causal_lm=causal_lm,
        overriding_architecture_config=config,
    )

    assert decoder.decoder.config.use_cache is False
    # Speech2Text reuses this configuration when loading its generation model.
    assert decoder.overriding_architecture_config["trust_remote_code"] is False
    assert overrides == {"use_cache": False, "trust_remote_code": True}


def test_default_architecture_overrides_disable_custom_code(local_hf_model):
    model_path, causal_lm = local_hf_model
    decoder = HuggingFaceTransformersDecoder(
        vocab_size=16,
        encoder_output_size=8,
        model_name_or_path=model_path,
        causal_lm=causal_lm,
    )
    assert decoder.overriding_architecture_config["trust_remote_code"] is False


@pytest.mark.parametrize("local_hf_model", [True], indirect=True, ids=["causal"])
def test_custom_tokenizer_code_is_disabled(local_hf_model, monkeypatch):
    model_path, _ = local_hf_model
    tokenizer_config_path = Path(model_path) / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text())
    tokenizer_config["tokenizer_class"] = "CustomTokenizer"
    tokenizer_config["auto_map"] = {
        "AutoTokenizer": ["custom_tokenizer.CustomTokenizer", None]
    }
    tokenizer_config_path.write_text(json.dumps(tokenizer_config))
    (Path(model_path) / "custom_tokenizer.py").write_text(
        'raise AssertionError("Custom tokenizer code was executed")\n'
    )

    def unexpected_prompt(*args):
        pytest.fail("Model loading must not prompt to execute custom code")

    monkeypatch.setattr("builtins.input", unexpected_prompt)
    with pytest.raises(ValueError, match="trust_remote_code"):
        HuggingFaceTransformersDecoder(
            vocab_size=16,
            encoder_output_size=8,
            model_name_or_path=model_path,
            causal_lm=True,
        )
