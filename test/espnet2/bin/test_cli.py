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

import numpy as np
import pytest
import soundfile

from espnet2.bin import cli


def _fake_module(monkeypatch, name, attr, obj):
    module = types.ModuleType(name)
    setattr(module, attr, obj)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class _Recorder:
    """Stands in for an inference class: records how it was built and called."""

    def __init__(self, result):
        self.result = result
        self.tag = None
        self.device = None
        self.calls = []
        self.fs = 16000

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


def test_asr_prints_the_transcript(monkeypatch, tmp_path, capsys):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    rec = _Recorder("hello there")
    _fake_module(monkeypatch, "espnet2.bin.s2t_inference", "Speech2Text", rec)

    assert cli.main(["asr", str(audio)]) == 0

    assert capsys.readouterr().out.strip() == "hello there"
    assert rec.tag == cli.DEFAULT_MODELS["asr"]
    assert rec.device == "cpu"
    args, kwargs = rec.calls[0]
    assert args == (str(audio),)
    # no language given means the model detects it
    assert kwargs == {"lang_sym": "<nolang>", "task_sym": "<asr>"}


def test_asr_passes_the_language_and_the_chosen_model(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    rec = _Recorder("x")
    _fake_module(monkeypatch, "espnet2.bin.s2t_inference", "Speech2Text", rec)

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
    assert rec.calls[0][1]["lang_sym"] == "<jpn>"


def test_translate_builds_the_target_token(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    rec = _Recorder("bonjour")
    _fake_module(monkeypatch, "espnet2.bin.s2t_inference", "Speech2Text", rec)

    assert cli.main(["translate", str(audio), "--to", "fra", "--language", "eng"]) == 0

    assert rec.calls[0][1] == {"lang_sym": "<eng>", "task_sym": "<st_fra>"}


def test_translate_requires_a_target():
    with pytest.raises(SystemExit) as e:
        cli.main(["translate", "a.wav"])
    assert e.value.code == 2  # argparse's "bad usage"


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
    for command in ("asr", "translate", "tts", "enhance", "models"):
        assert command in out


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

    class Mismatch:
        @staticmethod
        def from_pretrained(model_tag=None, device=None, **kwargs):
            raise TypeError("__init__() got an unexpected keyword argument 'x'")

    _fake_module(
        monkeypatch,
        "espnet2.bin.s2t_inference",
        "Speech2Text",
        Mismatch,
    )

    assert cli.main(["asr", str(audio), "--model", "espnet/a-tts-model"]) == 1

    err = capsys.readouterr().err
    assert "does not look like a model for `espnet asr`" in err
    assert "espnet models" in err


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
    for command in ("asr", "translate", "tts", "enhance", "models"):
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


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda", "cuda:1"])
def test_the_devices_torch_understands_are_accepted(device):
    assert cli._device(device) == device


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


def test_only_an_unexpected_keyword_is_blamed_on_the_model(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")

    class Bug:
        def __init__(self, s2t_train_config=None, **kwargs):
            pass

        @staticmethod
        def from_pretrained(model_tag=None, device=None, **kwargs):
            raise TypeError("unsupported operand type(s) for +: 'int' and 'str'")

    _fake_module(monkeypatch, "espnet2.bin.s2t_inference", "Speech2Text", Bug)

    # a TypeError from inside the model must not be reported as "wrong model"
    with pytest.raises(TypeError, match="unsupported operand"):
        cli.main(["asr", str(audio)])


def test_a_keyword_the_constructor_does_take_is_not_blamed_on_the_model(
    monkeypatch, tmp_path
):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")

    class Bug:
        def __init__(self, device=None, **kwargs):
            pass

        @staticmethod
        def from_pretrained(model_tag=None, device=None, **kwargs):
            # raised from deeper inside, about an argument this class accepts
            raise TypeError("f() got an unexpected keyword argument 'device'")

    _fake_module(monkeypatch, "espnet2.bin.s2t_inference", "Speech2Text", Bug)

    with pytest.raises(TypeError, match="'device'"):
        cli.main(["asr", str(audio)])
