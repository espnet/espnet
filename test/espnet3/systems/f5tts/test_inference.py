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

from espnet3.api.inference import Audio
from espnet3.systems.base.inference_runner import InferenceRunner
from espnet3.systems.f5tts.f5tts import F5TTS
from espnet3.systems.f5tts.inference import (
    F5TTSInference,
    Inference,
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
)


class _StubVocos:
    """Stands in for Vocos: exposes ``decode``, upsamples by the hop length."""

    def decode(self, mel):
        """Return silence one hop length long per mel frame."""
        return torch.zeros(1, mel.shape[-1] * 256)


# ------------------------------------------------------------------- _chunk_text


def test_chunk_text_keeps_short_text_whole():
    """Text under the budget stays one chunk."""
    assert _chunk_text("Hello there.", max_chars=100) == ["Hello there."]


def test_chunk_text_splits_on_sentence_boundaries():
    """Longer text is cut between sentences, each chunk within the budget."""
    chunks = _chunk_text("One. Two. Three.", max_chars=9)

    assert chunks == ["One. Two.", "Three."]
    assert all(len(c.encode("utf-8")) <= 9 for c in chunks)


def test_chunk_text_splits_full_width_punctuation():
    """The zh boundary class has no trailing space, so it splits differently."""
    assert _chunk_text("你好。世界。", max_chars=9) == ["你好。", "世界。"]


def test_chunk_text_of_empty_text_is_empty():
    """Empty text gives no chunks."""
    assert _chunk_text("", max_chars=10) == []


# -------------------------------------------------------------------- _cross_fade


def test_cross_fade_of_no_waves_returns_silence():
    """No waveforms give a single silent sample."""
    assert _cross_fade([], 0.1, 24000).shape == (1,)


def test_cross_fade_of_a_single_wave_is_a_passthrough():
    """A single waveform is returned as is."""
    wave = np.arange(10, dtype=np.float32)

    assert _cross_fade([wave], 0.1, 24000) is wave


def test_zero_duration_cross_fade_is_plain_concatenation():
    """A zero-length cross-fade concatenates the waveforms."""
    a = np.ones(4, dtype=np.float32)
    b = np.zeros(4, dtype=np.float32)

    np.testing.assert_array_equal(
        _cross_fade([a, b], 0.0, 24000), np.concatenate([a, b])
    )


def test_cross_fade_overlaps_and_shortens_the_result():
    """The overlap is shared, so the result is shorter and the level is kept."""
    a = np.ones(10, dtype=np.float32)
    b = np.ones(10, dtype=np.float32)
    n = 4  # 4 samples at sr=1000 is 0.004 s

    out = _cross_fade([a, b], 0.004, 1000)

    # The overlap is shared rather than appended, so the join costs n samples.
    assert len(out) == len(a) + len(b) - n
    # Two constant-1 ramps that sum to 1 leave the level untouched.
    np.testing.assert_allclose(out, np.ones(16, dtype=np.float32), atol=1e-6)


def test_cross_fade_falls_back_to_concatenation_when_a_wave_is_too_short():
    """An overlap that rounds to zero samples concatenates instead."""
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


def test_an_over_long_sentence_splits_at_word_boundaries():
    """Each chunk is spoken as its own utterance, so a split word is audible."""
    text = "the quick brown fox jumps over the lazy dog"

    chunks = _chunk_text(text, max_chars=16)

    # Every chunk must consist of whole words from the input.
    words = set(text.split())
    for chunk in chunks:
        assert set(chunk.split()) <= words, f"{chunk!r} contains a word fragment"
    assert " ".join(chunks) == text


def test_a_split_landing_on_a_space_does_not_strand_a_fragment():
    """The boundary is consumed as the separator, not left mid-word."""
    assert _chunk_text("abc de", max_chars=4) == ["abc", "de"]


def test_a_word_wider_than_the_budget_falls_back_to_a_character_split():
    """No whitespace to cut at, so the budget still has to be honoured."""
    chunks = _chunk_text("supercalifragilistic", max_chars=6)

    assert all(len(chunk.encode("utf-8")) <= 6 for chunk in chunks)
    assert "".join(chunks) == "supercalifragilistic"


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
            "_target_": "espnet3.systems.f5tts.f5tts.F5TTS",
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
    """Save the reference model's weights as a Lightning-style checkpoint."""
    path = tmp_path / "last.ckpt"
    torch.save({"state_dict": reference_model.state_dict()}, path)
    return path


@pytest.fixture
def stub_vocoder(monkeypatch):
    """Replace the Vocos download with :class:`_StubVocos`."""
    monkeypatch.setattr(
        F5TTSInference, "_load_vocoder", lambda self, path: _StubVocos()
    )


@pytest.fixture
def engine(train_config, checkpoint_path, stub_vocoder):
    """Build an engine on the tiny model with a stub vocoder and fixed seed."""
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


def test_sample_rate_defaults_to_the_models(engine):
    """Unset, the rate is the one the mel front end was configured with."""
    assert engine.target_sample_rate == FEATS_CONF["fs"]


def test_a_sample_rate_other_than_the_models_is_refused(
    train_config, checkpoint_path, stub_vocoder
):
    """A 16 kHz rate on a 24 kHz model would resample and mislabel the audio."""
    with pytest.raises(ValueError, match="target_sample_rate=16000"):
        F5TTSInference(
            train_config=str(train_config),
            checkpoint_path=str(checkpoint_path),
            target_sample_rate=16000,
        )
    # The model's own rate, given explicitly as older configs do, is accepted.
    engine = F5TTSInference(
        train_config=str(train_config),
        checkpoint_path=str(checkpoint_path),
        target_sample_rate=FEATS_CONF["fs"],
    )
    assert engine.target_sample_rate == FEATS_CONF["fs"]


def test_a_vocoder_at_another_rate_is_refused(
    monkeypatch, train_config, checkpoint_path
):
    """Vocos reports its training rate on its mel feature extractor."""
    vocoder = _StubVocos()
    vocoder.feature_extractor = types.SimpleNamespace(
        mel_spec=types.SimpleNamespace(sample_rate=22050)
    )
    monkeypatch.setattr(F5TTSInference, "_load_vocoder", lambda self, path: vocoder)

    with pytest.raises(ValueError, match="vocoder produces 22050 Hz"):
        F5TTSInference(
            train_config=str(train_config), checkpoint_path=str(checkpoint_path)
        )


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
    """With ``use_ema=False`` the raw weights are loaded, not the EMA ones."""
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
    """A training config without ``model._target_`` is refused."""
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"model": {"hidden_size": 32}}), encoding="utf-8")

    with pytest.raises(ValueError, match="model._target_"):
        F5TTSInference(train_config=str(path), checkpoint_path=str(checkpoint_path))


def test_a_config_without_a_token_list_is_rejected(
    tmp_path, train_config, checkpoint_path, stub_vocoder
):
    """A training config without a token list is refused."""
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
        "_target_": "espnet3.systems.f5tts.preprocessor.F5PinyinPreprocessor",
        "vocab_file": str(vocab),
    }
    path = tmp_path / "pinyin.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    engine = F5TTSInference(
        train_config=str(path), checkpoint_path=str(checkpoint_path)
    )

    # Built lazily, so this asserts the branch was taken, not pypinyin's output.
    assert callable(engine._tokenize)


# ----------------------------------------------------------------- generation


def test_infer_one_returns_a_waveform(engine):
    """``infer_one`` returns a non-empty mono float32 waveform."""
    wav = engine.infer_one(
        "abc", np.zeros(24000 // 2, dtype=np.float32), reference_text="ab"
    )

    assert wav.ndim == 1 and wav.dtype == np.float32
    assert len(wav) > 1


def test_infer_one_requires_the_reference_transcript(engine):
    """A missing transcript is refused, not replaced by the target text."""
    with pytest.raises(TypeError):
        engine.infer_one("abc", np.zeros(24000 // 2, dtype=np.float32))
    for empty in ("", "   "):
        with pytest.raises(ValueError, match="reference_text is required"):
            engine.infer_one("abc", np.zeros(24000 // 2, dtype=np.float32), empty)


def test_a_stereo_reference_is_downmixed(engine):
    """A two-channel reference is averaged to mono."""
    wav = engine.infer_one(
        "abc", np.zeros((2, 24000 // 2), dtype=np.float32), reference_text="ab"
    )

    assert wav.ndim == 1


def test_call_returns_a_wav_entry_for_a_single_sample(engine):
    """Calling the engine on one sample returns only a ``wav`` array."""
    out = engine(
        text="abc", speech=np.zeros(24000 // 2, dtype=np.float32), reference_text="ab"
    )

    assert set(out) == {"wav"}
    assert isinstance(out["wav"], np.ndarray)


def test_call_maps_over_a_batch(engine):
    """Lists of inputs give one waveform per item."""
    audio = [np.zeros(24000 // 2, dtype=np.float32)] * 2

    out = engine(
        text=["abc", "ba"], reference_speech=audio, reference_text=["ab", "ab"]
    )

    assert len(out["wav"]) == 2


def test_call_without_a_reference_is_refused(engine):
    """A call without reference audio is refused."""
    with pytest.raises(ValueError, match="No reference audio"):
        engine(text="abc", reference_text="ab")


def test_call_without_a_reference_transcript_is_refused(engine):
    """A call without the reference transcript is refused, single or batched."""
    audio = np.zeros(24000 // 2, dtype=np.float32)
    with pytest.raises(ValueError, match="No reference transcript"):
        engine(text="abc", reference_speech=audio)
    with pytest.raises(ValueError, match="No reference transcript"):
        engine(text=["abc", "ba"], reference_speech=[audio, audio])


# ------------------------------------------------------- vocoder construction
#
# These exercise the real ``_load_vocoder`` (the ``engine`` fixture stubs the
# whole method out) by standing a fake vocos package up in ``sys.modules``,
# since the real one would otherwise reach for the network.


class _FakeVocosModel:
    """Stands in for a loaded Vocos model: records the weights it is given."""

    def __init__(self):
        """Start with no weights loaded."""
        self.loaded_state = None

    def load_state_dict(self, state):
        """Remember the state dict it was given."""
        self.loaded_state = state

    def to(self, device):
        """Return itself, as a device move would."""
        return self

    def eval(self):
        """Return itself, as switching to eval mode would."""
        return self


def _install_fake_vocos(monkeypatch, created):
    """Put a fake ``vocos`` module in ``sys.modules`` that records its calls."""
    module = types.ModuleType("vocos")

    class Vocos:
        """Stands in for ``vocos.Vocos``: records how it was built."""

        @staticmethod
        def from_pretrained(repo):
            """Record the repository and return a fake model."""
            created["repo"] = repo
            return _FakeVocosModel()

        @staticmethod
        def from_hparams(config_path):
            """Record the config path and return a fake model."""
            created["config_path"] = config_path
            return _FakeVocosModel()

    module.Vocos = Vocos
    monkeypatch.setitem(sys.modules, "vocos", module)


def test_vocos_is_fetched_from_the_default_repo(
    monkeypatch, train_config, checkpoint_path
):
    """Without ``vocoder_path`` the default Vocos repository is used."""
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


# ------------------------------------------------------------ checkpoint loading


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


@pytest.fixture
def restore_g2p_registry():
    """register_f5_pinyin_g2p mutates espnet2 globals; undo it afterwards.

    It appends to ``g2p_choices``, replaces ``PhonemeTokenizer.__init__`` and
    flips the module-level ``_REGISTERED`` flag, so without this the tests that
    run after it see a patched tokenizer.
    """
    import espnet2.text.phoneme_tokenizer as pt
    from espnet3.systems.f5tts import pinyin

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


# ----------------------------------------------------- degenerate generation


def test_the_prompt_is_measured_with_the_mel_cfm_uses(engine):
    """samples // hop under-counts the centre-padded vocos front end by one.

    Slicing with the short value leaves the prompt's last frame at the head of
    the generated audio.
    """
    captured = {}
    real_sample = engine.cfm.sample

    def spy(cond, text, duration, **kwargs):
        """Record the requested duration and prompt length, then return silence."""
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
        """Return exactly the prompt's length, so nothing new is generated."""
        ref_len = cond.shape[-1] // engine.hop_length
        return torch.zeros(1, ref_len, 100), None

    monkeypatch.setattr(engine.cfm, "sample", prompt_only)

    wav = engine.infer_one(
        "abc", np.zeros(24000 // 2, dtype=np.float32), reference_text="ab"
    )

    np.testing.assert_array_equal(wav, np.zeros(1, dtype=np.float32))


# ------------------------------------------- Inference (espnet3.api.inference)


class _RecordingEngine:
    """Stands in for F5TTSInference: remembers what ``infer_one`` was given."""

    target_sample_rate = 24000

    def __init__(self):
        """Start with no recorded calls."""
        self.calls = []

    def infer_one(self, target_text, reference_audio, reference_text):
        """Record the arguments and return a short constant waveform."""
        self.calls.append((target_text, reference_audio, reference_text))
        return np.full(480, 0.25, dtype=np.float32)


def test_inference_declares_the_f5tts_fields():
    """``Inference`` declares three required inputs and one audio output."""
    assert [field.name for field in Inference.inputs] == [
        "text",
        "reference_speech",
        "reference_text",
    ]
    # All required: a missing transcript is refused, not guessed.
    assert [field.optional for field in Inference.inputs] == [False, False, False]
    assert [(field.name, field.kind) for field in Inference.outputs] == [
        ("wav", "audio")
    ]


def test_inference_returns_audio_at_the_vocoder_rate():
    """The output is an ``Audio`` at 24 kHz and the inputs reach the engine."""
    backend = _RecordingEngine()
    model = Inference(backend)

    output = model("hello", np.zeros(2400, dtype=np.float32), "a prompt")

    assert model.sample_rate == 24000
    assert isinstance(output["wav"], Audio)
    assert output["wav"].rate == 24000
    assert output["wav"].array.shape == (480,)
    text, reference_audio, reference_text = backend.calls[0]
    assert (text, reference_text) == ("hello", "a prompt")
    assert reference_audio.shape == (2400,)


def test_inference_resamples_what_gradio_hands_over():
    """A ``(rate, int16 samples)`` pair reaches the engine as float at 24 kHz."""
    backend = _RecordingEngine()
    model = Inference(backend)
    one_second_at_48k = (48000, np.full(48000, 16384, dtype=np.int16))

    model(text="hello", reference_speech=one_second_at_48k, reference_text="hi")

    _, reference_audio, reference_text = backend.calls[0]
    assert reference_audio.dtype == np.float32
    assert abs(len(reference_audio) - 24000) <= 1
    assert abs(float(np.median(reference_audio)) - 0.5) < 0.01
    assert reference_text == "hi"


def test_inference_downmixes_a_stereo_reference():
    """A stereo ``(rate, samples)`` reference reaches the engine as mono."""
    backend = _RecordingEngine()
    stereo = np.zeros((2400, 2), dtype=np.float32)

    Inference(backend)("hello", (24000, stereo), "a prompt")

    assert backend.calls[0][1].ndim == 1


def test_inference_requires_text_a_reference_and_its_transcript():
    """Each missing input is refused by name, an empty transcript included."""
    model = Inference(_RecordingEngine())
    audio = np.zeros(2400, dtype=np.float32)

    with pytest.raises(TypeError, match="reference_speech"):
        model("hello", reference_text="a prompt")
    with pytest.raises(TypeError, match="text"):
        model(reference_speech=audio, reference_text="a prompt")
    with pytest.raises(TypeError, match="reference_text"):
        model("hello", audio)
    # An empty box in the demo arrives as None, which counts as not given.
    with pytest.raises(TypeError, match="reference_text"):
        model("hello", audio, None)


def test_inference_runs_a_batch_item_by_item():
    """``batch`` runs the engine once per item, in order."""
    backend = _RecordingEngine()
    model = Inference(backend)
    reference = np.zeros(2400, dtype=np.float32)

    outputs = model.batch(
        [
            {"text": "first", "reference_speech": reference, "reference_text": "a"},
            {"text": "second", "reference_speech": reference, "reference_text": "b"},
        ]
    )

    assert [output["wav"].rate for output in outputs] == [24000, 24000]
    assert [call[0] for call in backend.calls] == ["first", "second"]
    assert [call[2] for call in backend.calls] == ["a", "b"]


def test_inference_builds_the_engine_from_its_own_arguments(
    train_config, checkpoint_path, stub_vocoder
):
    """What ``inference.yaml`` does: the class takes the engine's arguments."""
    model = Inference(
        train_config=str(train_config),
        checkpoint_path=str(checkpoint_path),
        device="cpu",
        ode_solver_steps=2,
        cross_fade_duration=0.0,
        seed=0,
    )

    assert isinstance(model.backend, F5TTSInference)
    output = model("abc", np.random.RandomState(0).randn(4800).astype(np.float32), "ab")
    assert output["wav"].array.dtype == np.float32
    assert output["wav"].array.ndim == 1


def test_inference_serves_the_infer_stage_runner(engine):
    """``InferenceRunner`` picks the declared inputs out of the item by name.

    No ``input_key`` is given: the declaration says what to read, the item's
    ``utt_id`` becomes the record's id, and undeclared columns are ignored.
    """
    model = Inference(engine)
    reference = np.random.RandomState(0).randn(4800).astype(np.float32)
    dataset = {
        0: {
            "utt_id": "u0",
            "text": "abc",
            "reference_speech": reference,
            "reference_text": "ab",
            "speaker": "ignored",
        }
    }

    output = InferenceRunner.forward(0, dataset=dataset, model=model)

    assert output["utt_id"] == "u0"
    assert output["wav"].rate == 24000
    assert "speaker" not in output
