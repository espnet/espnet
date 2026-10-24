"""ParallelLLM built for inference: from the config, without the base weights.

Inference loads a checkpoint strictly over the whole model, so the base LM's
pretrained weights would be overwritten before use. These tests build a tiny
real Qwen3 and check that the inference build never loads it, yet ends up
with the same parameters and the same rotary buffers as the training build.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from espnet2.speechlm.model.speechlm.multimodal_io.abs_io import AbsIO

transformers = pytest.importorskip("transformers")

VOCAB = 100


class _TextIO(AbsIO):
    def __init__(self):
        super().__init__(modality="text", is_discrete=True)

    def num_stream(self):
        return 1

    def get_vocabulary(self):
        return [f"t{i}" for i in range(VOCAB)]

    def get_stream_interval(self):
        return [(0, VOCAB)]

    def get_stream_weight(self):
        return [1.0]

    def preprocess(self, data):
        return np.zeros((1, 1)), None, np.zeros((1, 1))

    def find_length(self, data):
        return 1

    def copy_for_worker(self):
        return self


def _vocab_meta():
    special = 256
    size = special + VOCAB
    return {
        "vocab": [f"<|s{i}|>" for i in range(special)] + _TextIO().get_vocabulary(),
        "vocab_intervals": {
            "special_token": [(0, special)],
            "text": [(special, size)],
        },
        "vocab_weight": torch.ones(size),
        "vocab_size": size,
        "mm_start": None,
        "mm_end": None,
        "text_start": special,
        "text_end": size,
        "num_stream": 1,
    }


@pytest.fixture
def tiny_qwen3(tmp_path):
    """A real Qwen3 checkpoint directory, a few thousand parameters in size."""
    config = transformers.Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=VOCAB,
        architectures=["Qwen3ForCausalLM"],
    )
    torch.manual_seed(0)
    # saved in bfloat16, so the config records a dtype other than the default
    transformers.Qwen3ForCausalLM(config).to(torch.bfloat16).save_pretrained(tmp_path)
    return str(tmp_path)


def _build(path, is_train, monkeypatch, **options):
    from espnet2.speechlm.model.speechlm.lm.parallel import build_parallel_hf_class

    cls = build_parallel_hf_class(path)
    if not is_train:
        # the point of the inference build: the base weights are never read
        def refuse(*args, **kwargs):
            raise AssertionError("loaded the base model's weights")

        monkeypatch.setattr(transformers.Qwen3ForCausalLM, "from_pretrained", refuse)
    options = {"attn_implementation": "sdpa", "dtype": "float32", **options}
    return cls.from_pretrained(
        path,
        multimodal_io=nn.ModuleDict({"text": _TextIO()}),
        vocab_meta=_vocab_meta(),
        is_train=is_train,
        **options,
    )


def test_inference_build_never_loads_the_base_weights(tiny_qwen3, monkeypatch):
    model = _build(tiny_qwen3, False, monkeypatch)
    assert not any(t.is_meta for t in model.parameters())
    assert not any(t.is_meta for t in model.buffers())


def test_inference_build_has_what_a_checkpoint_fills(tiny_qwen3, monkeypatch):
    """The base model's own parameters, shapes and rotary table, unchanged.

    The embeddings and head are rebuilt for the multimodal vocabulary in
    both builds, so they are compared against that size instead.
    """
    base = transformers.Qwen3ForCausalLM.from_pretrained(tiny_qwen3)
    built = _build(tiny_qwen3, False, monkeypatch)
    rebuilt = {"model.embed_tokens.weight", "lm_head.weight"}

    built_state = built.state_dict()
    for name, tensor in base.state_dict().items():
        if name in rebuilt:
            assert built_state[name].shape == (_vocab_meta()["vocab_size"], 32)
        else:
            assert built_state[name].shape == tensor.shape, name
    built_buffers = dict(built.named_buffers())
    for name, buffer in base.named_buffers():
        torch.testing.assert_close(built_buffers[name], buffer, msg=name)


def test_weights_loaded_over_the_inference_build_compute_the_same(
    tiny_qwen3, monkeypatch
):
    """What inference does: once the weights are in, it is the same network."""
    base = transformers.Qwen3ForCausalLM.from_pretrained(
        tiny_qwen3, attn_implementation="sdpa", dtype="float32"
    ).eval()
    built = _build(tiny_qwen3, False, monkeypatch).eval()
    stack = {
        name: tensor
        for name, tensor in base.state_dict().items()
        if name.startswith("model.") and name != "model.embed_tokens.weight"
    }
    missing = set(built.load_state_dict(stack, strict=False).missing_keys)
    assert not {name for name in missing if name in stack}

    torch.manual_seed(1)
    embeds = torch.randn(1, 5, 32)
    with torch.no_grad():
        expected = base.model(inputs_embeds=embeds).last_hidden_state
        actual = built.model(inputs_embeds=embeds).last_hidden_state
    torch.testing.assert_close(actual, expected)


# ---------------------------------------------------------------------------
# The options a model_conf can carry, which from_pretrained accepts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "options",
    [
        {"local_files_only": True},
        {"revision": "main"},
        {"local_files_only": True, "revision": "main", "cache_dir": None},
        # weight-reading options have nothing to act on, and must not fail
        {"low_cpu_mem_usage": True, "use_safetensors": True},
    ],
    ids=["local_files_only", "revision", "hub-options", "weight-options"],
)
def test_inference_build_takes_the_hub_options(tiny_qwen3, monkeypatch, options):
    model = _build(tiny_qwen3, False, monkeypatch, **options)
    assert not any(t.is_meta for t in model.parameters())


def test_hub_options_reach_the_config(tiny_qwen3, monkeypatch):
    """The config is read under the same revision and offline rules."""
    seen = {}
    original = transformers.AutoConfig.from_pretrained

    def recording(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", recording)
    from espnet2.speechlm.model.speechlm.lm import parallel

    monkeypatch.setattr(parallel.AutoConfig, "from_pretrained", recording)
    _build(tiny_qwen3, False, monkeypatch, local_files_only=True, revision="main")
    assert seen["local_files_only"] is True
    assert seen["revision"] == "main"


@pytest.mark.parametrize(
    "dtype_option",
    [{"dtype": "auto"}, {"dtype": None}, {"torch_dtype": "auto"}],
    ids=["auto", "unset", "torch_dtype-auto"],
)
def test_auto_dtype_is_the_one_from_pretrained_picks(
    tiny_qwen3, monkeypatch, dtype_option
):
    """A dtype of "auto", or none at all, means the one the config records."""
    expected = transformers.Qwen3ForCausalLM.from_pretrained(
        tiny_qwen3, dtype="auto"
    ).dtype
    assert expected == torch.bfloat16  # what the fixture saved
    options = {"dtype": None, **dtype_option}
    if options["dtype"] is None:
        options.pop("dtype")
    from espnet2.speechlm.model.speechlm.lm.parallel import build_parallel_hf_class

    cls = build_parallel_hf_class(tiny_qwen3)
    monkeypatch.setattr(
        transformers.Qwen3ForCausalLM,
        "from_pretrained",
        lambda *a, **k: pytest.fail("loaded the base model's weights"),
    )
    model = cls.from_pretrained(
        tiny_qwen3,
        multimodal_io=nn.ModuleDict({"text": _TextIO()}),
        vocab_meta=_vocab_meta(),
        is_train=False,
        attn_implementation="sdpa",
        **options,
    )
    assert model.model.layers[0].self_attn.q_proj.weight.dtype == expected
