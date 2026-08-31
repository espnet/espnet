"""The recipe-facing inference engine, ``F5TTSInference``.

Every test here is offline: the model is rebuilt from a tiny training config
written into ``tmp_path``, the checkpoint is one this module saves, and the
vocoder is stubbed. Nothing downloads.
"""

import logging
import sys
import types

import numpy as np
import pytest
import torch
import yaml

from espnet3.systems.tts.f5_tts.f5tts import F5TTS
from espnet3.systems.tts.f5_tts.inference import (
    F5TTSInference,
    _chunk_text,
    _cross_fade,
)

TOKENS = ["<blank>", "<unk>", "a", "b", "c", " ", "<sos/eos>"]
MODEL_CONF = dict(
    hidden_size=32,
    depth=1,
    attention_heads=2,
    attention_head_size=16,
    feed_forward_multiplier=1,
    text_embedding_size=16,
    convolution_layers=1,
    ode_solver_method="euler",
)
FEATS_CONF = dict(
    fs=24000,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=100,
    mel_spec_type="vocos",
)


class _StubVocos:
    """Stands in for Vocos: exposes ``decode``, upsamples by the hop length."""

    def decode(self, mel):
        return torch.zeros(1, mel.shape[-1] * 256)


class _StubBigVGAN:
    """Stands in for BigVGAN: a plain callable with no ``decode``."""

    def __call__(self, mel):
        return torch.zeros(1, mel.shape[-1] * 256)


# ------------------------------------------------------------------- _chunk_text


def test_chunk_text_keeps_short_text_whole():
    assert _chunk_text("Hello there.", max_chars=100) == ["Hello there."]


def test_chunk_text_splits_on_sentence_boundaries():
    chunks = _chunk_text("One. Two. Three.", max_chars=9)

    assert chunks == ["One. Two.", "Three."]
    assert all(len(c.encode("utf-8")) <= 9 for c in chunks)


def test_chunk_text_splits_full_width_punctuation():
    """The zh boundary class has no trailing space, so it splits differently."""
    assert _chunk_text("你好。世界。", max_chars=9) == ["你好。", "世界。"]


def test_chunk_text_of_empty_text_is_empty():
    assert _chunk_text("", max_chars=10) == []


# -------------------------------------------------------------------- _cross_fade


def test_cross_fade_of_no_waves_returns_silence():
    assert _cross_fade([], 0.1, 24000).shape == (1,)


def test_cross_fade_of_a_single_wave_is_a_passthrough():
    wave = np.arange(10, dtype=np.float32)

    assert _cross_fade([wave], 0.1, 24000) is wave


def test_zero_duration_cross_fade_is_plain_concatenation():
    a = np.ones(4, dtype=np.float32)
    b = np.zeros(4, dtype=np.float32)

    np.testing.assert_array_equal(
        _cross_fade([a, b], 0.0, 24000), np.concatenate([a, b])
    )


def test_cross_fade_overlaps_and_shortens_the_result():
    a = np.ones(10, dtype=np.float32)
    b = np.ones(10, dtype=np.float32)
    n = 4  # 4 samples at sr=1000 is 0.004 s

    out = _cross_fade([a, b], 0.004, 1000)

    # The overlap is shared rather than appended, so the join costs n samples.
    assert len(out) == len(a) + len(b) - n
    # Two constant-1 ramps that sum to 1 leave the level untouched.
    np.testing.assert_allclose(out, np.ones(16, dtype=np.float32), atol=1e-6)


def test_cross_fade_falls_back_to_concatenation_when_a_wave_is_too_short():
    a = np.ones(3, dtype=np.float32)
    b = np.ones(3, dtype=np.float32)

    # 0.1 s at sr=1 rounds the overlap down to 0 samples.
    assert len(_cross_fade([a, b], 0.1, 1)) == 6


def test_a_sentence_longer_than_the_budget_is_split():
    """No internal punctuation must not mean an unbounded chunk."""
    text = "word " * 200  # 1000 bytes, nothing for the sentence splitter to use

    chunks = _chunk_text(text, max_chars=100)

    assert len(chunks) > 1
    assert all(len(chunk.encode("utf-8")) <= 100 for chunk in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_an_over_long_cjk_sentence_splits_on_character_boundaries():
    """Cutting mid-character would corrupt the text before it is tokenized."""
    text = "字" * 300  # 900 bytes, 3 bytes per character

    chunks = _chunk_text(text, max_chars=90)

    assert all(len(chunk.encode("utf-8")) <= 90 for chunk in chunks)
    assert "".join(chunks) == text
    for chunk in chunks:
        chunk.encode("utf-8").decode("utf-8")  # would raise on a split character


def test_a_single_character_wider_than_the_budget_is_still_emitted():
    """Degenerate budget: there is nothing smaller to cut to."""
    assert _chunk_text("字", max_chars=1) == ["字"]


# ------------------------------------------------------------------- fixtures


@pytest.fixture
def train_config(tmp_path):
    """A minimal training YAML of the shape the recipe writes."""
    token_file = tmp_path / "tokens.txt"
    token_file.write_text("\n".join(TOKENS) + "\n", encoding="utf-8")
    cfg = {
        "model": {
            "_target_": "espnet3.systems.tts.f5_tts.f5tts.F5TTS",
            "token_list": str(token_file),
            "feats_extract_config": dict(FEATS_CONF),
            **dict(MODEL_CONF),
        },
        "dataset": {
            "preprocessor": {
                "_target_": "espnet2.train.preprocessor.CommonPreprocessor",
                "token_type": "char",
                "token_list": list(TOKENS),
            }
        },
    }
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture
def reference_model(train_config):
    """The same architecture the engine will rebuild from ``train_config``."""
    cfg = yaml.safe_load(train_config.read_text(encoding="utf-8"))["model"]
    return F5TTS(
        token_list=cfg["token_list"],
        feats_extract_config=cfg["feats_extract_config"],
        **MODEL_CONF,
    )


@pytest.fixture
def checkpoint_path(tmp_path, reference_model):
    path = tmp_path / "last.ckpt"
    torch.save({"state_dict": reference_model.state_dict()}, path)
    return path


@pytest.fixture
def stub_vocoder(monkeypatch):
    monkeypatch.setattr(
        F5TTSInference, "_load_vocoder", lambda self, name, path: _StubVocos()
    )


@pytest.fixture
def engine(train_config, checkpoint_path, stub_vocoder):
    return F5TTSInference(
        train_config=str(train_config),
        checkpoint_path=str(checkpoint_path),
        ode_solver_steps=2,
        cross_fade_duration=0.0,
        seed=0,
    )


# --------------------------------------------------------------- construction


def test_construction_wires_up_the_model_parts(engine):
    """The engine holds the pieces generation needs, not the wrapper alone."""
    assert engine.cfm is engine.model.cfm
    assert engine.feats_extract is engine.model.feats_extract
    # hop_length is read from the config rather than assumed.
    assert engine.hop_length == 256


def test_checkpoint_weights_are_actually_loaded(engine, reference_model):
    """A silent load failure would leave random weights behind."""
    loaded = dict(engine.model.state_dict())
    for key, expected in reference_model.state_dict().items():
        torch.testing.assert_close(loaded[key], expected)


def test_ema_weights_are_preferred_when_present(
    tmp_path, train_config, reference_model, stub_vocoder
):
    """Training saves EMA weights under their own prefixed key."""
    ema = {
        "ema_model." + k: torch.zeros_like(v)
        for k, v in reference_model.state_dict().items()
    }
    path = tmp_path / "ema.ckpt"
    torch.save(
        {"state_dict": reference_model.state_dict(), "ema_model_state_dict": ema}, path
    )

    engine = F5TTSInference(train_config=str(train_config), checkpoint_path=str(path))

    # The EMA copy is all zeros, so picking it up is unambiguous.
    for value in engine.model.state_dict().values():
        if value.is_floating_point():
            assert torch.all(value == 0)


def test_ema_is_skipped_when_use_ema_is_off(
    tmp_path, train_config, reference_model, stub_vocoder
):
    ema = {
        "ema_model." + k: torch.zeros_like(v)
        for k, v in reference_model.state_dict().items()
    }
    path = tmp_path / "ema.ckpt"
    torch.save(
        {"state_dict": reference_model.state_dict(), "ema_model_state_dict": ema}, path
    )

    engine = F5TTSInference(
        train_config=str(train_config), checkpoint_path=str(path), use_ema=False
    )

    loaded = dict(engine.model.state_dict())
    for key, expected in reference_model.state_dict().items():
        torch.testing.assert_close(loaded[key], expected)


def test_a_config_without_a_model_target_is_rejected(tmp_path, checkpoint_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"model": {"hidden_size": 32}}), encoding="utf-8")

    with pytest.raises(ValueError, match="model._target_"):
        F5TTSInference(train_config=str(path), checkpoint_path=str(checkpoint_path))


def test_a_config_without_a_token_list_is_rejected(
    tmp_path, train_config, checkpoint_path, stub_vocoder
):
    cfg = yaml.safe_load(train_config.read_text(encoding="utf-8"))
    del cfg["dataset"]["preprocessor"]["token_list"]
    path = tmp_path / "no_tokens.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="token_list"):
        F5TTSInference(train_config=str(path), checkpoint_path=str(checkpoint_path))


def test_a_vocab_file_selects_the_pinyin_tokenizer(
    tmp_path, train_config, checkpoint_path, stub_vocoder
):
    """``vocab_file`` routes to F5's own pinyin vocab instead of espnet2's."""
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(TOKENS) + "\n", encoding="utf-8")
    cfg = yaml.safe_load(train_config.read_text(encoding="utf-8"))
    cfg["dataset"]["preprocessor"] = {
        "_target_": "espnet3.systems.tts.f5_tts.preprocessor.F5PinyinPreprocessor",
        "vocab_file": str(vocab),
    }
    path = tmp_path / "pinyin.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    engine = F5TTSInference(
        train_config=str(path), checkpoint_path=str(checkpoint_path)
    )

    # Built lazily, so this asserts the branch was taken, not pypinyin's output.
    assert callable(engine._tokenize)


# ------------------------------------------------------------------- vocoder


def test_an_unknown_vocoder_name_is_rejected(train_config, checkpoint_path):
    with pytest.raises(ValueError, match="Unsupported vocoder"):
        F5TTSInference(
            train_config=str(train_config),
            checkpoint_path=str(checkpoint_path),
            vocoder_name="griffin_lim",
        )


def test_vocoders_without_decode_are_called_directly(engine):
    """Vocos exposes ``decode``; BigVGAN is a plain module."""
    engine.vocoder = _StubBigVGAN()

    assert engine._vocode(torch.zeros(1, 100, 3)).shape == (3 * 256,)


# ----------------------------------------------------------------- native F5


def test_native_f5_state_is_flattened_to_cfm_level(tmp_path, reference_model):
    """The official checkpoint nests EMA weights and adds bookkeeping tensors."""
    cfm_state_dict = reference_model.cfm.state_dict()
    raw = {"ema_model." + k: v for k, v in cfm_state_dict.items()}
    raw["initted"] = torch.tensor(True)
    raw["step"] = torch.tensor(7)
    path = tmp_path / "model.pt"
    torch.save({"ema_model_state_dict": raw}, path)

    out = F5TTSInference._load_native_f5_state(str(path), use_ema=True)

    assert set(out) == set(cfm_state_dict)
    assert "initted" not in out and "step" not in out


def test_native_f5_falls_back_to_the_non_ema_state(tmp_path, reference_model):
    cfm_state_dict = reference_model.cfm.state_dict()
    path = tmp_path / "model.pt"
    torch.save({"model_state_dict": dict(cfm_state_dict)}, path)

    out = F5TTSInference._load_native_f5_state(str(path), use_ema=True)

    assert set(out) == set(cfm_state_dict)


# ----------------------------------------------------------------- generation


def test_infer_one_returns_a_waveform(engine):
    wav = engine.infer_one(
        "abc", np.zeros(24000 // 2, dtype=np.float32), reference_text="ab"
    )

    assert wav.ndim == 1 and wav.dtype == np.float32
    assert len(wav) > 1


def test_ref_text_defaults_to_the_target_text(engine):
    """Self-reference: no transcript given, so the target doubles as one."""
    wav = engine.infer_one("abc", np.zeros(24000 // 2, dtype=np.float32))

    assert wav.ndim == 1


def test_a_stereo_reference_is_downmixed(engine):
    wav = engine.infer_one(
        "abc", np.zeros((2, 24000 // 2), dtype=np.float32), reference_text="ab"
    )

    assert wav.ndim == 1


def test_call_returns_a_wav_entry_for_a_single_sample(engine):
    out = engine(text="abc", speech=np.zeros(24000 // 2, dtype=np.float32))

    assert set(out) == {"wav"}
    assert isinstance(out["wav"], np.ndarray)


def test_call_maps_over_a_batch(engine):
    audio = [np.zeros(24000 // 2, dtype=np.float32)] * 2

    out = engine(
        text=["abc", "ba"], reference_speech=audio, reference_text=["ab", "ab"]
    )

    assert len(out["wav"]) == 2


def test_call_without_a_reference_is_refused(engine):
    with pytest.raises(ValueError, match="No reference audio"):
        engine(text="abc")


# ------------------------------------------------------- vocoder construction
#
# These exercise the real ``_load_vocoder`` (the ``engine`` fixture stubs the
# whole method out) by standing fake vocoder packages up in ``sys.modules``.
# Neither vocos nor bigvgan is an espnet dependency, and both would otherwise
# reach for the network.


class _FakeVocosModel:
    def __init__(self):
        self.loaded_state = None

    def load_state_dict(self, state):
        self.loaded_state = state

    def to(self, device):
        return self

    def eval(self):
        return self


def _install_fake_vocos(monkeypatch, created):
    module = types.ModuleType("vocos")

    class Vocos:
        @staticmethod
        def from_pretrained(repo):
            created["repo"] = repo
            return _FakeVocosModel()

        @staticmethod
        def from_hparams(config_path):
            created["config_path"] = config_path
            return _FakeVocosModel()

    module.Vocos = Vocos
    monkeypatch.setitem(sys.modules, "vocos", module)


def test_vocos_is_fetched_from_the_default_repo(
    monkeypatch, train_config, checkpoint_path
):
    created = {}
    _install_fake_vocos(monkeypatch, created)

    F5TTSInference(train_config=str(train_config), checkpoint_path=str(checkpoint_path))

    assert created["repo"] == "charactr/vocos-mel-24khz"


def test_a_local_vocoder_path_is_loaded_from_disk(
    monkeypatch, tmp_path, train_config, checkpoint_path
):
    """An offline recipe points at a checkout instead of the hub."""
    created = {}
    _install_fake_vocos(monkeypatch, created)
    vocoder_dir = tmp_path / "vocos"
    vocoder_dir.mkdir()
    (vocoder_dir / "config.yaml").write_text("{}", encoding="utf-8")
    torch.save({"weight": torch.zeros(1)}, vocoder_dir / "pytorch_model.bin")

    engine = F5TTSInference(
        train_config=str(train_config),
        checkpoint_path=str(checkpoint_path),
        vocoder_path=str(vocoder_dir),
    )

    assert created["config_path"] == f"{vocoder_dir}/config.yaml"
    assert "repo" not in created  # the hub was not consulted
    assert engine.vocoder.loaded_state is not None


def test_bigvgan_weight_norm_is_removed_at_load(
    monkeypatch, train_config, checkpoint_path
):
    """BigVGAN must be switched to its inference form before sampling."""
    calls = []
    module = types.ModuleType("bigvgan")

    class _FakeBigVGAN:
        def remove_weight_norm(self):
            calls.append("remove_weight_norm")

        def to(self, device):
            return self

        def eval(self):
            return self

    class BigVGAN:
        @staticmethod
        def from_pretrained(repo, use_cuda_kernel=False):
            calls.append(repo)
            return _FakeBigVGAN()

    module.BigVGAN = BigVGAN
    monkeypatch.setitem(sys.modules, "bigvgan", module)

    F5TTSInference(
        train_config=str(train_config),
        checkpoint_path=str(checkpoint_path),
        vocoder_name="bigvgan",
    )

    assert calls == ["nvidia/bigvgan_v2_24khz_100band_256x", "remove_weight_norm"]


def test_a_missing_vocoder_package_is_reported_clearly(
    monkeypatch, train_config, checkpoint_path
):
    """vocos is optional, so the failure must name the install."""
    monkeypatch.setitem(sys.modules, "vocos", None)

    with pytest.raises(ImportError, match="pip install vocos"):
        F5TTSInference(
            train_config=str(train_config), checkpoint_path=str(checkpoint_path)
        )


# ------------------------------------------------- native F5 checkpoint loading


def test_a_native_f5_checkpoint_loads_into_the_cfm(
    tmp_path, train_config, reference_model, stub_vocoder
):
    """Official SWivid weights sit at CFM level, below the espnet model."""
    cfm_state_dict = reference_model.cfm.state_dict()
    zeros = {"ema_model." + k: torch.zeros_like(v) for k, v in cfm_state_dict.items()}
    path = tmp_path / "native.pt"
    torch.save({"ema_model_state_dict": zeros}, path)

    engine = F5TTSInference(
        train_config=str(train_config), checkpoint_path=str(path), native_f5=True
    )

    for value in engine.cfm.state_dict().values():
        if value.is_floating_point():
            assert torch.all(value == 0)


def test_a_partial_checkpoint_still_loads(
    tmp_path, train_config, reference_model, stub_vocoder, caplog
):
    """strict=False, so a key mismatch is a warning rather than a crash."""
    state = dict(reference_model.state_dict())
    state.pop(next(iter(state)))
    state["not_a_real_parameter"] = torch.zeros(1)
    path = tmp_path / "partial.ckpt"
    torch.save({"state_dict": state}, path)

    with caplog.at_level(logging.WARNING):
        F5TTSInference(train_config=str(train_config), checkpoint_path=str(path))

    assert "missing keys" in caplog.text
    assert "unexpected keys" in caplog.text


# ------------------------------------------------------ remaining load paths


def test_a_safetensors_checkpoint_is_read_as_a_flat_ema_dict(tmp_path, reference_model):
    """The official release ships .safetensors with no nesting."""
    safetensors_torch = pytest.importorskip("safetensors.torch")
    cfm_state_dict = reference_model.cfm.state_dict()
    flat = {"ema_model." + k: v.contiguous() for k, v in cfm_state_dict.items()}
    flat["initted"] = torch.tensor(True)
    path = tmp_path / "model.safetensors"
    safetensors_torch.save_file(flat, str(path))

    out = F5TTSInference._load_native_f5_state(str(path), use_ema=True)

    assert set(out) == set(cfm_state_dict)


def test_a_bare_state_dict_checkpoint_is_used_as_is(tmp_path, reference_model):
    """Neither ema_model_state_dict nor model_state_dict: take the whole file."""
    cfm_state_dict = reference_model.cfm.state_dict()
    path = tmp_path / "bare.pt"
    torch.save(dict(cfm_state_dict), path)

    out = F5TTSInference._load_native_f5_state(str(path), use_ema=True)

    assert set(out) == set(cfm_state_dict)


@pytest.fixture
def restore_g2p_registry():
    """register_f5_pinyin_g2p mutates espnet2 globals; undo it afterwards.

    It appends to ``g2p_choices``, replaces ``PhonemeTokenizer.__init__`` and
    flips the module-level ``_REGISTERED`` flag, so without this the tests that
    run after it see a patched tokenizer.
    """
    import espnet2.text.phoneme_tokenizer as pt
    from espnet3.systems.tts.f5_tts import pinyin

    choices = list(pt.g2p_choices)
    init = pt.PhonemeTokenizer.__init__
    registered = pinyin._REGISTERED
    yield
    pt.g2p_choices[:] = choices
    pt.PhonemeTokenizer.__init__ = init
    pinyin._REGISTERED = registered


def test_the_f5_pinyin_g2p_is_registered_when_the_config_asks_for_it(
    tmp_path, train_config, checkpoint_path, stub_vocoder, restore_g2p_registry
):
    """g2p_type: f5_pinyin has to be patched into espnet2 before use."""
    import espnet2.text.phoneme_tokenizer as pt

    cfg = yaml.safe_load(train_config.read_text(encoding="utf-8"))
    cfg["dataset"]["preprocessor"]["g2p_type"] = "f5_pinyin"
    cfg["dataset"]["preprocessor"]["token_type"] = "phn"
    path = tmp_path / "g2p.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    F5TTSInference(train_config=str(path), checkpoint_path=str(checkpoint_path))

    assert "f5_pinyin" in pt.g2p_choices


def test_a_missing_bigvgan_package_is_reported_clearly(
    monkeypatch, train_config, checkpoint_path
):
    monkeypatch.setitem(sys.modules, "bigvgan", None)

    with pytest.raises(ImportError, match="bigvgan is required"):
        F5TTSInference(
            train_config=str(train_config),
            checkpoint_path=str(checkpoint_path),
            vocoder_name="bigvgan",
        )


# ----------------------------------------------------- degenerate generation


def test_the_prompt_is_measured_with_the_mel_cfm_uses(engine):
    """samples // hop under-counts the centre-padded vocos front end by one.

    Slicing with the short value leaves the prompt's last frame at the head of
    the generated audio.
    """
    captured = {}
    real_sample = engine.cfm.sample

    def spy(cond, text, duration, **kwargs):
        captured["duration"] = duration
        captured["prompt_frames"] = engine.cfm.mel_spec(cond).shape[-1]
        return torch.zeros(1, duration, 100), None

    engine.cfm.sample = spy
    try:
        wav = engine.infer_one(
            "abc", np.random.randn(8000).astype(np.float32), reference_text="ab"
        )
    finally:
        engine.cfm.sample = real_sample

    # The stub vocoder upsamples one mel frame to 256 samples.
    generated_frames = len(wav) // 256
    assert generated_frames == captured["duration"] - captured["prompt_frames"]
    assert captured["prompt_frames"] == 8000 // 256 + 1  # not 8000 // 256


def test_a_silent_reference_does_not_produce_nan(engine):
    """rms == 0 would make target_rms / rms divide by zero and NaN everything."""
    wav = engine.infer_one(
        "abc", np.zeros(24000 // 2, dtype=np.float32), reference_text="ab"
    )

    assert not np.isnan(wav).any()


def test_mismatched_batch_lengths_are_rejected(engine):
    """zip would truncate silently and misalign outputs with test samples."""
    with pytest.raises(ValueError, match="matching lengths"):
        engine(
            text=["a", "b", "c"],
            reference_speech=[np.zeros(1200, dtype=np.float32)] * 2,
            reference_text=["x", "y"],
        )


def test_empty_target_text_returns_silence(engine):
    """Nothing to say, so there are no chunks to synthesize."""
    wav = engine.infer_one(
        "", np.zeros(24000 // 2, dtype=np.float32), reference_text="ab"
    )

    np.testing.assert_array_equal(wav, np.zeros(1, dtype=np.float32))


def test_a_chunk_that_generates_no_frames_is_dropped(engine, monkeypatch):
    """If the solver returns only the prompt there is nothing left to vocode."""

    def prompt_only(cond, text, duration, **kwargs):
        # Return exactly the reference length, so the generated span is empty.
        ref_len = cond.shape[-1] // engine.hop_length
        return torch.zeros(1, ref_len, 100), None

    monkeypatch.setattr(engine.cfm, "sample", prompt_only)

    wav = engine.infer_one(
        "abc", np.zeros(24000 // 2, dtype=np.float32), reference_text="ab"
    )

    np.testing.assert_array_equal(wav, np.zeros(1, dtype=np.float32))
