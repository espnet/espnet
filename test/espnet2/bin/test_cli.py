"""The `espnet` command: dispatch, defaults and the errors a user can fix.

No model is downloaded here. Each test replaces the inference class the
subcommand imports, so what is checked is the wiring: which class is asked
for which tag, what it is called with, and what reaches the terminal.
"""

import importlib.metadata
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import soundfile

from espnet2.bin import cli


def _fake_module(monkeypatch, name, attr, obj):
    module = types.ModuleType(name)
    setattr(module, attr, obj)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class _FakeModel:
    """Stands in for an esp2_s2t Inference: records what it was asked."""

    def __init__(self, result, error=None):
        self.result = result
        self.error = error  # what the Inference raises before decoding, if anything
        self.tag = self.system = self.device = None
        self.calls = []

    def __call__(self, audio, **prompt):
        if self.error is not None:
            raise self.error
        self.calls.append((audio, prompt))
        return {"text": self.result, "segments": []}


def _fake_load(monkeypatch, model):
    """Make espnet3.api.inference.load hand out `model`, recording the call."""
    import espnet3.api.inference as api

    def load(tag, *, system=None, device="auto", **kwargs):
        model.tag, model.system, model.device = tag, system, device
        return model

    monkeypatch.setattr(api, "load", load)
    return model


class _Recorder:
    """Stands in for an ESPnet2 inference class: records how it was built and called."""

    def __init__(self, result, ctc_only=True, nolang="<nolang>", tokens=None):
        self.result = result
        self.tag = None
        self.device = None
        self.calls = []
        self.fs = 16000
        # what `espnet phonemize` reads off a loaded checkpoint: whether there is
        # a decoder to segment long audio with, and how this model spells
        # "work the language out yourself"
        self.ctc_only = ctc_only
        self.preprocessor_conf = {"speech_length": 20, "fs": 16000}
        if nolang is not None:
            self.preprocessor_conf["nolang_symbol"] = nolang
        self.s2t_model = types.SimpleNamespace(
            token_list=list(tokens) if tokens is not None else None
        )

    def read_audio(self, path):
        # two windows of the length above, so that a test can see both
        return np.zeros(20 * 16000 * 2, dtype=np.float32)

    def no_language(self):
        # Speech2Text.no_language, which the command line only rewords: the
        # config's symbol, then either spelling, each checked against the
        # token list
        tokens = set(self.s2t_model.token_list or ())
        for candidate in (
            self.preprocessor_conf.get("nolang_symbol"),
            "<nolang>",
            "<unk>",
        ):
            if candidate and (not tokens or candidate in tokens):
                return candidate
        raise ValueError("this model has no symbol for an unknown language")

    def from_pretrained(self, model_tag=None, device=None, **kwargs):
        self.tag, self.device = model_tag, device
        return self

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result

    def best_path(self, *args, **kwargs):
        # what --live and --stream call: one window, decoded on the CTC head
        # with no search, because the next window is already arriving
        self.calls.append((args, kwargs))
        return self.result

    def decode_long(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        # one (start, end, text) per segment: two of them, so that a test
        # reading the printed line sees them joined rather than concatenated
        first, _, rest = self.result.partition(" ")
        if not rest:
            return [(0.0, 1.0, self.result)]
        return [(0.0, 1.0, first), (1.0, 2.0, rest)]


def test_models_lists_one_default_per_command(capsys):
    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out
    for task, tag in cli.DEFAULT_MODELS.items():
        assert task in out and tag in out


def test_every_default_names_a_model_in_the_espnet_organisation():
    # a typo here would only show as a download failure on a user's machine
    for task, tag in cli.DEFAULT_MODELS.items():
        assert tag.startswith("espnet/"), (task, tag)


def test_transcribe_prints_the_transcript(monkeypatch, tmp_path, capsys):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    model = _fake_load(monkeypatch, _FakeModel("hello there"))

    assert cli.main(["transcribe", str(audio)]) == 0

    assert capsys.readouterr().out.strip() == "hello there"
    # the default model, through the s2t system, --device auto resolved
    # before the command ran (no GPU here: cpu)
    assert (model.tag, model.system, model.device) == (
        cli.DEFAULT_MODELS["transcribe"],
        "s2t",
        "cpu",
    )
    # transcription asked for by name; no language given means the model works it out
    assert model.calls == [(str(audio), {"task": "asr"})]


def test_asr_passes_the_language_and_the_chosen_model(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    rec = _fake_load(monkeypatch, _FakeModel("x"))
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)  # so cuda passes
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    assert (
        cli.main(
            [
                "asr",
                str(audio),
                "--language",
                "jpn",
                "--model",
                "espnet/other",
                "--device",
                "cuda",
            ]
        )
        == 0
    )

    assert rec.tag == "espnet/other" and rec.device == "cuda"
    assert rec.calls[0][1] == {"task": "asr", "language": "jpn"}


def test_phonemize_prints_ipa_without_the_slashes(monkeypatch, tmp_path, capsys):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    # POWSM writes one phone between each pair of slashes
    rec = _fake_load(monkeypatch, _FakeModel("/ð//ə//s//e//ɪ/"))

    assert cli.main(["phonemize", str(audio)]) == 0

    assert capsys.readouterr().out.strip() == "ðəseɪ"
    assert rec.tag == cli.DEFAULT_MODELS["phonemize"]
    assert rec.calls == [(str(audio), {"task": "pr"})]


def test_phonemize_can_print_one_phone_at_a_time(monkeypatch, tmp_path, capsys):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    # the point of the slashes: pʰ is one phone, not p followed by ʰ
    _fake_load(monkeypatch, _FakeModel("/pʰ//ɔ//s//ə//m/"))

    assert cli.main(["phonemize", str(audio), "--spaced"]) == 0

    assert capsys.readouterr().out.strip() == "pʰ ɔ s ə m"


def test_phonemize_passes_a_language_when_one_is_given(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    rec = _fake_load(monkeypatch, _FakeModel("/a/"))

    assert cli.main(["phonemize", str(audio), "--language", "jpn"]) == 0

    assert rec.calls[0][1] == {"task": "pr", "language": "jpn"}


def test_phonemize_prints_what_a_model_said_when_there_are_no_slashes(
    monkeypatch, tmp_path, capsys
):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    _fake_load(monkeypatch, _FakeModel("hello there"))

    assert cli.main(["phonemize", str(audio)]) == 0

    # swallowing it would be worse than printing a transcript nobody wanted
    assert capsys.readouterr().out.strip() == "hello there"


class _FakeSegment:
    """What ForcedAligner returns: an utterance, when it was said, and how sure.

    A token has the same four fields, which is what lets the command print
    both with one line of formatting.
    """

    def __init__(self, text, start, end, score=0.9, tokens=None):
        self.text, self.start, self.end, self.score = text, start, end, score
        self.tokens = (
            tokens
            if tokens is not None
            else [_FakeSegment(text.split()[0], start, start + 0.1, 0.8, tokens=[])]
        )


class _FakeAligner:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def from_pretrained(self, model_tag=None, device=None, **kwargs):
        self.tag, self.device = model_tag, device
        return self

    def __call__(self, speech, utterances):
        self.calls.append((speech, list(utterances)))
        return [
            _FakeSegment(text, i * 1.0, i * 1.0 + 0.5)
            for i, text in enumerate(utterances)
        ]


def _fake_aligner(monkeypatch, aligner=None):
    aligner = aligner or _FakeAligner()
    _fake_module(monkeypatch, "espnet2.bin.align", "ForcedAligner", aligner)
    return aligner


def test_align_prints_a_line_an_utterance(monkeypatch, tmp_path, capsys):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    rec = _fake_aligner(monkeypatch)

    assert cli.main(["align", str(audio), "--text", "one", "--text", "two"]) == 0

    lines = capsys.readouterr().out.strip().split("\n")
    assert [line.split("\t")[-1] for line in lines] == ["one", "two"]
    assert lines[0].startswith("0.00\t0.50\t")
    assert rec.tag == cli.DEFAULT_MODELS["align"]
    assert rec.calls == [(str(audio), ["one", "two"])]


def test_align_can_print_each_token(monkeypatch, tmp_path, capsys):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    _fake_aligner(monkeypatch)

    assert cli.main(["align", str(audio), "--text", "one two", "--tokens"]) == 0

    lines = capsys.readouterr().out.strip().split("\n")
    # the utterance, then the tokens it was made of, indented
    assert len(lines) == 2 and lines[1].startswith("  ")


def test_align_reads_a_file_of_utterances(monkeypatch, tmp_path, capsys):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    utterances = tmp_path / "utts.txt"
    utterances.write_text("one\n\ntwo\n")  # blank lines are not utterances
    rec = _fake_aligner(monkeypatch)

    assert cli.main(["align", str(audio), "--text-file", str(utterances)]) == 0

    assert rec.calls[0][1] == ["one", "two"]


def test_align_passes_the_device_it_was_given(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    rec = _fake_aligner(monkeypatch)
    import torch

    # a machine with an MPS device, so that the name passes the check in main()
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert cli.main(["align", str(audio), "--text", "one", "--device", "mps"]) == 0

    assert rec.device == "mps"


def test_align_reports_text_that_cannot_fit(monkeypatch, tmp_path, capsys):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")

    class _TooMuch(_FakeAligner):
        def __call__(self, speech, utterances):
            raise ValueError("120 tokens to align against 40 frames")

    _fake_aligner(monkeypatch, _TooMuch())

    assert cli.main(["align", str(audio), "--text", "one"]) == 1

    # the aligner's own sentence, not a traceback
    assert "120 tokens to align against 40 frames" in capsys.readouterr().err


def test_align_needs_text_and_only_one_way_of_giving_it(monkeypatch, tmp_path, capsys):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")

    assert cli.main(["align", str(audio)]) == 1
    assert "--text" in capsys.readouterr().err

    assert cli.main(["align", str(audio), "--text", "one", "--text-file", "f.txt"]) == 1
    assert "not both" in capsys.readouterr().err


def test_translate_builds_the_target_token(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    rec = _fake_load(monkeypatch, _FakeModel("bonjour"))

    assert cli.main(["translate", str(audio), "--to", "fra", "--language", "eng"]) == 0

    assert rec.calls[0][1] == {"task": "st_fra", "language": "eng"}


def test_translate_requires_a_target():
    with pytest.raises(SystemExit) as e:
        cli.main(["translate", "a.wav"])
    assert e.value.code == 2  # argparse's "bad usage"


OWSM_TOKENS = ["<blank>", "<unk>", "<na>", "<nolang>", "<eng>", "<jpn>", "<asr>"]
OWSM_TOKENS += ["<st_deu>", "<st_eng>", "<st_jpn>"]


@pytest.mark.parametrize(
    "argv",
    [
        ["transcribe", "--language", "en"],
        ["translate", "--language", "en", "--to", "eng"],
        ["phonemize", "--language", "en"],
    ],
)
def test_a_language_the_model_lacks_is_a_sentence_not_a_traceback(
    monkeypatch, tmp_path, capsys, argv
):
    # the esp2_s2t Inference checks the symbol before decoding and raises a
    # ValueError that names the fix; the command line prints it and stops
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    rec = _fake_load(
        monkeypatch,
        _FakeModel(
            "x",
            error=ValueError("language 'en': this model has no <en>; pass ISO 639-3"),
        ),
    )

    assert cli.main([argv[0], str(audio), *argv[1:]]) == 1

    err = capsys.readouterr().err
    assert "'en'" in err and "ISO 639-3" in err
    assert rec.calls == []  # nothing was decoded with a symbol it lacks


def test_stream_checks_the_language_before_it_starts(monkeypatch, tmp_path, capsys):
    said = []
    _stub_live(monkeypatch, [np.zeros(16000, dtype=np.float32)], said)
    rec = _Recorder([("x", ["x"], [1], "x", None)], tokens=OWSM_TOKENS)
    _fake_module(monkeypatch, "espnet2.bin.s2t_inference", "Speech2Text", rec)
    audio = tmp_path / "a.wav"
    soundfile.write(audio, np.zeros(16000, dtype=np.float32), 16000)

    assert cli.main(["transcribe", str(audio), "--stream", "--language", "en"]) == 1

    assert said == [] and rec.calls == []
    assert "'en'" in capsys.readouterr().err


def test_tts_writes_a_wave(monkeypatch, tmp_path):
    import numpy as np
    import torch

    rec = _Recorder({"wav": torch.from_numpy(np.zeros(160, dtype=np.float32))})
    _fake_module(monkeypatch, "espnet2.bin.tts_inference", "Text2Speech", rec)
    out = tmp_path / "said.wav"

    assert cli.main(["tts", "hello", "-o", str(out)]) == 0

    assert out.exists() and out.stat().st_size > 0
    assert rec.calls[0][0] == ("hello",)


def test_enhance_writes_one_file_per_speaker(monkeypatch, tmp_path):
    import numpy as np
    import soundfile as sf

    source = tmp_path / "mix.wav"
    sf.write(source, np.zeros(160, dtype=np.float32), 16000)
    rec = _Recorder([np.zeros((1, 160), dtype=np.float32) for _ in range(2)])
    _fake_module(monkeypatch, "espnet2.bin.enh_inference", "SeparateSpeech", rec)
    out = tmp_path / "clean.wav"

    assert cli.main(["enhance", str(source), "-o", str(out)]) == 0

    assert (tmp_path / "clean.spk1.wav").exists()
    assert (tmp_path / "clean.spk2.wav").exists()
    assert not out.exists()


def test_enhance_writes_one_file_when_there_is_one_output(monkeypatch, tmp_path):
    import numpy as np
    import soundfile as sf

    source = tmp_path / "mix.wav"
    sf.write(source, np.zeros(160, dtype=np.float32), 16000)
    rec = _Recorder([np.zeros((1, 160), dtype=np.float32)])
    _fake_module(monkeypatch, "espnet2.bin.enh_inference", "SeparateSpeech", rec)
    out = tmp_path / "clean.wav"

    assert cli.main(["enhance", str(source), "-o", str(out)]) == 0

    assert out.exists()


@pytest.mark.parametrize("command", ["asr", "translate", "enhance"])
def test_a_missing_file_fails_before_a_model_is_fetched(command, capsys, monkeypatch):
    def explode(*a, **k):  # pragma: no cover - the point is that it is not reached
        raise AssertionError("downloaded a model for a file that is not there")

    monkeypatch.setattr(cli, "_load_audio", explode)
    argv = [command, "/nowhere/missing.wav"]
    if command == "translate":
        argv += ["--to", "eng"]

    assert cli.main(argv) == 1

    assert "no such file" in capsys.readouterr().err


def _stub_live(monkeypatch, blocks, said):
    """Replace the module `cmd_asr` imports, so nothing is recorded or decoded."""
    live = types.ModuleType("espnet2.bin.live")
    live.LiveError = type("LiveError", (RuntimeError,), {})
    live.from_file = lambda path, **kw: iter(blocks)
    live.from_microphone = lambda **kw: iter(blocks)

    def transcribe(decode, source, **kwargs):
        for chunk in source:
            said.append(decode(chunk))
        return 0

    live.transcribe = transcribe
    monkeypatch.setitem(sys.modules, "espnet2.bin.live", live)
    # `from espnet2.bin import live` reads the package attribute first, and
    # another test may already have imported the real module
    import espnet2.bin

    monkeypatch.setattr(espnet2.bin, "live", live, raising=False)
    return live


def test_stream_decodes_a_file_window_by_window(monkeypatch, tmp_path):
    said = []
    _stub_live(monkeypatch, [np.zeros(16000, dtype=np.float32)], said)
    recorder = _Recorder([("hello", ["h"], [1], "hello", None)])
    _fake_module(
        monkeypatch,
        "espnet2.bin.s2t_inference",
        "Speech2Text",
        recorder,
    )
    audio = tmp_path / "a.wav"
    soundfile.write(audio, np.zeros(16000, dtype=np.float32), 16000)

    assert cli.main(["asr", str(audio), "--stream"]) == 0
    assert said == ["hello"]


def test_live_records_instead_of_reading_a_file(monkeypatch):
    said = []
    _stub_live(monkeypatch, [np.zeros(16000, dtype=np.float32)], said)
    recorder = _Recorder([("spoken", ["s"], [1], "spoken", None)])
    _fake_module(
        monkeypatch,
        "espnet2.bin.s2t_inference",
        "Speech2Text",
        recorder,
    )

    assert cli.main(["asr", "--live"]) == 0
    assert said == ["spoken"]


def test_live_with_a_file_is_refused(monkeypatch, capsys, tmp_path):
    _stub_live(monkeypatch, [], [])
    audio = tmp_path / "a.wav"
    soundfile.write(audio, np.zeros(16000, dtype=np.float32), 16000)
    assert cli.main(["asr", str(audio), "--live"]) == 1
    assert "do not also name a file" in capsys.readouterr().err


def test_stream_without_a_file_is_refused(monkeypatch, capsys):
    _stub_live(monkeypatch, [], [])
    assert cli.main(["asr", "--stream"]) == 1
    assert "--stream needs an audio file" in capsys.readouterr().err


def test_a_recording_failure_is_reported_like_the_others(monkeypatch, capsys):
    live = _stub_live(monkeypatch, [], [])

    def refuse(**kwargs):
        raise live.LiveError("no microphone here")

    live.from_microphone = refuse
    recorder = _Recorder([])
    _fake_module(
        monkeypatch,
        "espnet2.bin.s2t_inference",
        "Speech2Text",
        recorder,
    )
    assert cli.main(["asr", "--live"]) == 1
    err = capsys.readouterr().err
    assert err.strip().endswith("no microphone here") and "Traceback" not in err


def test_an_empty_result_becomes_an_empty_transcript(monkeypatch):
    said = []
    _stub_live(monkeypatch, [np.zeros(16000, dtype=np.float32)], said)
    recorder = _Recorder([])  # the model returned nothing for this window
    _fake_module(
        monkeypatch,
        "espnet2.bin.s2t_inference",
        "Speech2Text",
        recorder,
    )
    assert cli.main(["asr", "--live"]) == 0
    assert said == [""]


def test_the_help_lists_every_command(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for command in (
        "transcribe",
        "phonemize",
        "align",
        "translate",
        "synthesize",
        "enhance",
        "run",
        "demo",
        "models",
    ):
        assert command in out
    # and the names they had in a release, where someone looking for the one
    # they remember will look
    assert "(asr)" in out and "(tts)" in out


def test_an_output_without_an_extension_is_refused(monkeypatch, tmp_path, capsys):
    import numpy as np
    import soundfile as sf

    source = tmp_path / "mix.wav"
    sf.write(source, np.zeros(160, dtype=np.float32), 16000)

    def explode(*a, **k):  # pragma: no cover - not reached
        raise AssertionError("loaded a model before checking the output path")

    monkeypatch.setattr(cli, "_load_audio", explode)

    assert cli.main(["enhance", str(source), "-o", str(tmp_path / "clean")]) == 1

    assert "file extension" in capsys.readouterr().err


def test_an_output_soundfile_cannot_write_is_refused(monkeypatch, capsys):
    # this used to reach soundfile after the model had been downloaded and
    # run, and came out as a traceback
    import torch

    recorder = _Recorder({"wav": torch.zeros(160)})
    _fake_module(monkeypatch, "espnet2.bin.tts_inference", "Text2Speech", recorder)
    assert cli.main(["tts", "hello", "-o", "notes.txt"]) == 1
    err = capsys.readouterr().err
    assert "cannot write .txt audio" in err and "Traceback" not in err
    assert recorder.tag is None  # nothing was downloaded


def test_a_tag_for_another_task_is_explained(monkeypatch, tmp_path, capsys):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")

    import espnet3.api.inference as api
    from espnet2.utils.pretrained import ModelTagError

    def load(tag, **kwargs):
        raise ModelTagError(f"{tag} was published with tts_train_config")

    monkeypatch.setattr(api, "load", load)

    assert cli.main(["transcribe", str(audio), "--model", "espnet/a-tts-model"]) == 1

    err = capsys.readouterr().err
    assert "does not look like a model for `espnet transcribe`" in err
    assert "espnet models" in err and "tts_train_config" in err


def test_the_name_a_release_had_still_works(monkeypatch, tmp_path, capsys):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    _fake_load(monkeypatch, _FakeModel("hello there"))

    assert cli.main(["asr", str(audio)]) == 0

    printed = capsys.readouterr()
    assert printed.out.strip() == "hello there"
    # and it says where the name went, once, on stderr, so a pipe is unharmed
    assert "`asr` is now `transcribe`" in printed.err


def test_the_new_name_says_nothing_about_the_old_one(monkeypatch, tmp_path, capsys):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    _fake_load(monkeypatch, _FakeModel("hello there"))

    assert cli.main(["transcribe", str(audio)]) == 0

    assert capsys.readouterr().err == ""


def test_both_names_reach_the_same_command(monkeypatch, tmp_path, capsys):
    import torch

    out = tmp_path / "out.wav"
    rec = _Recorder({"wav": torch.from_numpy(np.zeros(160, dtype=np.float32))})
    _fake_module(monkeypatch, "espnet2.bin.tts_inference", "Text2Speech", rec)

    assert cli.main(["tts", "hello", "-o", str(out)]) == 0
    assert cli.main(["synthesize", "hello", "-o", str(out)]) == 0

    # one parser, two names: the same model and the same call
    assert rec.tag == cli.DEFAULT_MODELS["synthesize"]
    assert len(rec.calls) == 2 and rec.calls[0] == rec.calls[1]


def test_multichannel_audio_keeps_its_channels(tmp_path):
    import numpy as np
    import soundfile as sf

    source = tmp_path / "multi.wav"
    sf.write(source, np.zeros((160, 3), dtype=np.float32), 16000)

    speech, rate = cli._load_audio(str(source))

    # a beamformer needs every channel; mixing down here would silently
    # reduce it to one
    assert speech.shape == (160, 3)
    assert rate == 16000


# --- the command as a user runs it: a real process, no model downloaded ---


def _run(*args):
    """Invoke the console script's module the way the installed command does."""
    return subprocess.run(
        [sys.executable, "-m", "espnet2.bin.cli", *args],
        capture_output=True,
        text=True,
        cwd=Path(cli.__file__).parents[2],
    )


def test_help_runs_as_a_process_and_exits_zero():
    r = _run("--help")
    assert r.returncode == 0
    for command in ("asr", "translate", "tts", "enhance", "demo", "models"):
        assert command in r.stdout


def test_version_exits_zero_in_process(capsys):
    # the subprocess tests below do not count towards coverage, and this is
    # the line argparse runs when the flag is given
    with pytest.raises(SystemExit) as exit_:
        cli.main(["--version"])
    assert exit_.value.code == 0
    assert capsys.readouterr().out.strip() == f"espnet {cli._version()}"


def test_version_names_the_installed_version():
    r = _run("--version")
    assert r.returncode == 0
    assert r.stdout.strip() == f"espnet {cli._version()}"
    assert cli._version().strip() != ""


def test_version_says_so_when_the_package_is_not_installed(monkeypatch):
    # running from a source tree: nothing declares a version, and the command
    # still has to answer
    def missing(_name):
        raise importlib.metadata.PackageNotFoundError("espnet")

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert "source tree" in cli._version()


def test_version_does_not_need_a_subcommand():
    # --version is answered before the required subcommand is missed
    assert _run("--version").stderr == ""


def test_models_runs_as_a_process_and_names_the_defaults():
    r = _run("models")
    assert r.returncode == 0
    for tag in cli.DEFAULT_MODELS.values():
        assert tag in r.stdout


def test_a_user_error_is_one_line_and_exits_one():
    r = _run("asr", "/nowhere/missing.wav")
    assert r.returncode == 1
    assert r.stdout == ""
    assert r.stderr.strip().endswith("no such file: /nowhere/missing.wav")
    assert "Traceback" not in r.stderr


def test_an_unknown_subcommand_exits_two():
    r = _run("frobnicate")
    assert r.returncode == 2
    assert "invalid choice" in r.stderr


def test_an_unknown_device_is_refused_by_the_parser():
    r = _run("asr", "a.wav", "--device", "banana")
    assert r.returncode == 2
    assert "unknown device" in r.stderr


@pytest.mark.parametrize("device", ["auto", "cpu", "mps", "cuda", "cuda:1"])
def test_the_devices_torch_understands_are_accepted(device):
    assert cli._device(device) == device


@pytest.mark.parametrize("command", ["transcribe", "synthesize"])
def test_a_device_this_machine_lacks_is_a_sentence_before_the_download(
    monkeypatch, tmp_path, capsys, command
):
    """The name parses; whether torch has it is found when the model loads."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    _fake_load(monkeypatch, _FakeModel("x"))  # never reached
    _fake_module(monkeypatch, "espnet2.bin.tts_inference", "Text2Speech", _Recorder({}))
    arg = str(audio) if command == "transcribe" else "hello"

    assert cli.main([command, arg, "--device", "cuda:0"]) == 1

    err = capsys.readouterr().err
    assert err.startswith("espnet: cuda:0 asked for, but torch sees no CUDA device")


def test_a_write_failure_is_reported_like_the_others(monkeypatch, tmp_path, capsys):
    import numpy as np
    import torch

    rec = _Recorder({"wav": torch.from_numpy(np.zeros(160, dtype=np.float32))})
    _fake_module(monkeypatch, "espnet2.bin.tts_inference", "Text2Speech", rec)

    assert cli.main(["tts", "hello", "-o", str(tmp_path / "gone" / "x.wav")]) == 1

    assert "cannot write" in capsys.readouterr().err


def test_tts_checks_the_output_before_fetching_a_model(monkeypatch, capsys):
    def explode(*a, **k):  # pragma: no cover - not reached
        raise AssertionError("fetched a model before checking the output path")

    monkeypatch.setattr(cli, "_build", explode)

    assert cli.main(["tts", "hello", "-o", "out"]) == 1

    assert "file extension" in capsys.readouterr().err


# --- `espnet demo`: the browser app, built but never launched for real ---


class _FakeOWSM:
    """An OWSM-CTC stand-in: a token list to read menus from, and decoding."""

    def __init__(self, decoded="<eng><asr> hello there"):
        self.decoded = decoded
        self.tag = None
        self.device = None
        self.calls = []
        self.long_calls = []
        self.s2t_model = types.SimpleNamespace(
            token_list=[
                "<unk>",
                "<nolang>",
                "<eng>",
                "<jpn>",
                "<asr>",
                "<st_deu>",
                "<sos>",
            ]
        )

    preprocessor_conf = {"speech_length": 30, "nolang_symbol": "<nolang>"}
    # OWSM-CTC: read off the CTC head, no decoder to search over
    ctc_only = True

    def from_pretrained(self, model_tag=None, device=None, **kwargs):
        self.tag, self.device = model_tag, device
        return self

    def no_language(self):
        return self.preprocessor_conf["nolang_symbol"]

    def decode_window(self, speech, lang_sym=None, task_sym=None, text_prev="<na>"):
        # Speech2Text.decode_window: the checkpoint chooses its own way, and
        # this one is CTC-only. text_prev is what a task with a written input
        # is given - POWSM's <g2p> and <p2g> - and what primes the search on a
        # checkpoint that has one; this fake ignores it, as a CTC head does.
        return self.best_path(speech, lang_sym=lang_sym, task_sym=task_sym)[0][0]

    def best_path(self, speech, *args, **kwargs):
        # what the page calls for a single window: the CTC head, no search
        self.calls.append((speech, kwargs))
        return [(self.decoded,)]

    def decode_long(self, speech, **kwargs):
        self.long_calls.append((speech, kwargs))
        # (start, end, text) per segment, joined by the page
        return [(0.0, 15.0, "long form"), (15.0, 30.0, "text")]


def _fake_demo(monkeypatch, s2t=None, device="cpu", task="s2t"):
    """Wire up a gradio that records instead of serving, and a fake model."""
    import espnet

    gradio = mock.MagicMock()
    monkeypatch.setitem(sys.modules, "gradio", gradio)
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: device == "cuda")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    # the task check is one Hub request; the tests answer it themselves
    monkeypatch.setattr(espnet, "_infer_task", lambda tag: task)
    s2t = s2t or _FakeOWSM()
    _fake_module(monkeypatch, "espnet2.bin.s2t_inference", "Speech2Text", s2t)
    return gradio, s2t


def _launch(gradio):
    """How `app.launch(...)` was called on the Blocks the command built."""
    return gradio.Blocks.return_value.launch.call_args


def test_demo_serves_the_default_model_and_prints_the_url(monkeypatch, capsys):
    gradio, s2t = _fake_demo(monkeypatch)

    assert cli.main(["demo"]) == 0

    assert s2t.tag == cli.DEFAULT_MODELS["demo"] and s2t.device == "cpu"
    assert _launch(gradio).kwargs == {"server_port": 7860, "share": False}
    # the URL has to be on stdout before launch(), which blocks until Ctrl-C
    assert "http://127.0.0.1:7860" in capsys.readouterr().out


def test_demo_takes_the_port_the_share_flag_and_the_model(monkeypatch, capsys):
    gradio, s2t = _fake_demo(monkeypatch)

    argv = ["demo", "--model", "espnet/other", "--port", "8000", "--share"]
    assert cli.main(argv) == 0

    assert s2t.tag == "espnet/other"
    assert _launch(gradio).kwargs == {"server_port": 8000, "share": True}
    assert "http://127.0.0.1:8000" in capsys.readouterr().out


def test_demo_uses_the_gpu_when_there_is_one_and_the_flag_when_given(monkeypatch):
    gradio, s2t = _fake_demo(monkeypatch, device="cuda")

    assert cli.main(["demo"]) == 0
    assert s2t.device == "cuda"  # no --device: auto, and torch sees a GPU

    gradio, s2t = _fake_demo(monkeypatch, device="cuda")
    assert cli.main(["demo", "--device", "cpu"]) == 0
    assert s2t.device == "cpu"  # --device wins over the rule


def test_demo_without_gradio_is_a_user_error(monkeypatch, capsys):
    from espnet2.bin import demo

    def explode(*a, **k):  # pragma: no cover - not reached
        raise AssertionError("fetched a model before checking for gradio")

    monkeypatch.setattr(demo, "load_gradio", lambda: None)
    monkeypatch.setattr(cli, "_build", explode)

    assert cli.main(["demo"]) == 1

    err = capsys.readouterr().err
    assert err.startswith("espnet: gradio is not installed")
    assert "espnet[demo]" in err
    assert "Traceback" not in err


def test_demo_menus_come_from_the_checkpoint(monkeypatch):
    gradio, _ = _fake_demo(monkeypatch)

    assert cli.main(["demo"]) == 0

    # the two dropdowns are this checkpoint's own tokens, not a fixed list
    languages, targets = [call.args[0] for call in gradio.Dropdown.call_args_list]
    assert languages == ["Detect automatically", "English (eng)", "Japanese (jpn)"]
    # each task named by the abbreviation its field uses, then glossed
    assert targets == ["ASR: transcribe", "ST: translate to German (deu)"]


def _predict(gradio):
    """The function the Run button was wired to."""
    return gradio.Button.return_value.click.call_args.args[0]


def test_the_demo_decodes_a_short_recording(monkeypatch):
    import numpy as np

    from espnet2.bin import demo

    gradio, s2t = _fake_demo(monkeypatch)
    monkeypatch.setattr(
        demo, "read_audio", lambda path, rate=16000: np.zeros(16000 * 5, "float32")
    )
    assert cli.main(["demo"]) == 0

    language, text = _predict(gradio)("a.wav", demo.DETECT, demo.ASR_LABEL, False)

    assert (language, text) == ("English", "hello there")
    speech, kwargs = s2t.calls[0]
    # padded to the 30 s window OWSM is trained on, and the language left open
    assert len(speech) == 16000 * demo.WINDOW_SECS
    assert kwargs == {"lang_sym": "<nolang>", "task_sym": "<asr>"}


def test_the_demo_trims_audio_past_the_cap_and_says_so(monkeypatch):
    import numpy as np

    from espnet2.bin import demo

    gradio, s2t = _fake_demo(monkeypatch)
    long_audio = np.zeros(16000 * (demo.MAX_SECS + 30), "float32")
    monkeypatch.setattr(demo, "read_audio", lambda path, rate=16000: long_audio)
    assert cli.main(["demo"]) == 0

    _predict(gradio)("a.wav", "English (eng)", demo.ASR_LABEL, True)

    warned = " ".join(str(call) for call in gradio.Warning.call_args_list)
    assert f"first {demo.MAX_SECS} s" in warned
    speech, kwargs = s2t.long_calls[0]
    assert len(speech) == 16000 * demo.MAX_SECS
    # the language the user chose, so no detection pass was needed
    assert kwargs["lang_sym"] == "<eng>" and s2t.calls == []


def test_demo_refuses_a_model_for_another_task_before_downloading_it(
    monkeypatch, capsys
):
    gradio, s2t = _fake_demo(monkeypatch, task="tts")

    assert cli.main(["demo", "--model", "espnet/kan-bayashi_ljspeech_vits"]) == 1

    # refused on the metadata alone: nothing was built, so nothing was fetched
    assert s2t.tag is None
    err = capsys.readouterr().err
    assert "serves speech-to-text models" in err and "is a tts model" in err
    assert "Traceback" not in err


def test_demo_runs_when_the_metadata_says_nothing_about_the_task(monkeypatch):
    # the check turns a knowable mistake into a sentence; it is not a second
    # gate a valid checkpoint has to pass
    import espnet

    gradio, s2t = _fake_demo(monkeypatch)

    def unknown(tag):
        raise ValueError("cannot tell what task this is for")

    monkeypatch.setattr(espnet, "_infer_task", unknown)

    assert cli.main(["demo", "--model", "espnet/undocumented"]) == 0
    assert s2t.tag == "espnet/undocumented"


# --- `espnet run`: any model, by its declaration ---


class _Declared(_FakeModel):
    """A fake Inference with a declaration, as `espnet run` reads it."""

    def __init__(self, result):
        from espnet3.api.inference import Field

        super().__init__(result)
        self.inputs = (
            Field("speech", "audio", "Speech"),
            Field("language", "text", "Language", optional=True),
        )
        self.outputs = (
            Field("text", "text", "Transcription"),
            Field("segments", "segments", "Segments"),
        )

    def __call__(self, *args, **prompt):
        self.calls.append(prompt)
        return self.result


def _declared_model(monkeypatch, result=None):
    result = result or {
        "text": "hello world",
        "segments": [{"text": "hello world", "start": 0.0, "end": 1.25}],
    }
    return _fake_load(monkeypatch, _Declared(result))


def test_run_takes_inputs_by_name_and_prints_each_output(monkeypatch, capsys):
    model = _declared_model(monkeypatch)

    argv = [
        "run",
        "espnet/some_owsm",
        "--system",
        "s2t",
        "speech=a.wav",
        "language=jpn",
    ]
    assert cli.main(argv) == 0

    assert (model.tag, model.system) == ("espnet/some_owsm", "s2t")
    assert model.calls == [{"speech": "a.wav", "language": "jpn"}]
    assert (
        capsys.readouterr().out
        == "== text\nhello world\n== segments\n0.00\t1.25\thello world\n"
    )


def test_run_lists_the_declaration(monkeypatch, capsys):
    _declared_model(monkeypatch)

    assert cli.main(["run", "espnet/some_owsm", "--list"]) == 0

    assert capsys.readouterr().out == (
        "inputs:\n  speech=<audio>\n  language=<text>  (optional)\n"
        "outputs:\n  text  (text)\n  segments  (segments)\n"
    )


def test_run_refuses_an_input_the_model_does_not_declare(monkeypatch, capsys):
    model = _declared_model(monkeypatch)

    assert cli.main(["run", "espnet/some_owsm", "speech=a.wav", "beam=4"]) == 1

    assert (
        "'beam=4': give inputs as name=value, one of speech, language"
        in capsys.readouterr().err
    )
    assert model.calls == []


def test_run_names_a_required_input_left_out(monkeypatch, capsys):
    _declared_model(monkeypatch)

    assert cli.main(["run", "espnet/some_owsm", "language=jpn"]) == 1

    err = capsys.readouterr().err
    assert "missing input(s) ['speech']" in err and "speech=<audio>" in err


def test_run_writes_an_audio_output_as_a_wave(monkeypatch, tmp_path, capsys):
    from espnet3.api.inference import Audio, Field

    model = _declared_model(
        monkeypatch, {"audio": Audio(np.zeros(800, dtype=np.float32), 8000)}
    )
    model.outputs = (Field("audio", "audio", "Rendered"),)

    assert cli.main(["run", "x", "speech=a.wav", "-o", str(tmp_path / "out")]) == 0

    written = tmp_path / "out" / "audio.wav"
    assert capsys.readouterr().out.strip() == str(written)
    data, rate = soundfile.read(written)
    assert rate == 8000 and len(data) == 800


def test_run_reports_a_value_the_model_refuses(monkeypatch, capsys):
    _declared_model(monkeypatch)

    def refuse(self, *args, **prompt):
        raise ValueError("language 'xx': this model has no <xx>")

    monkeypatch.setattr(_Declared, "__call__", refuse)

    assert cli.main(["run", "x", "speech=a.wav", "language=xx"]) == 1
    assert "this model has no <xx>" in capsys.readouterr().err


def test_run_takes_an_input_after_an_option_and_other_commands_do_not(
    monkeypatch, capsys
):
    model = _declared_model(monkeypatch)

    # an input after an option is left over by argparse on every Python; an
    # input list that was empty before the option is filled by 3.12.7+ itself
    argv = ["run", "x", "speech=a.wav", "--system", "s2t", "language=jpn"]
    assert cli.main(argv) == 0
    assert model.calls == [{"speech": "a.wav", "language": "jpn"}]

    with pytest.raises(SystemExit) as e:
        cli.main(["models", "speech=a.wav"])
    assert e.value.code == 2
    assert "unrecognized arguments: speech=a.wav" in capsys.readouterr().err


def test_run_reports_a_tag_load_cannot_serve(monkeypatch, capsys):
    import espnet3.api.inference as api
    from espnet2.utils.pretrained import ModelTagError

    def load(tag, **kwargs):
        raise ModelTagError(f"{tag} is not a pack_model bundle")

    monkeypatch.setattr(api, "load", load)

    assert cli.main(["run", "espnet/nope", "speech=a.wav"]) == 1
    assert "not a pack_model bundle" in capsys.readouterr().err
