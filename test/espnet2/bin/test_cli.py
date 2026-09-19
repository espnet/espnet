"""The `espnet` command: dispatch, defaults and the errors a user can fix.

No model is downloaded here. Each test replaces the inference class the
subcommand imports, so what is checked is the wiring: which class is asked
for which tag, what it is called with, and what reaches the terminal.
"""

import subprocess
import sys
import types
from pathlib import Path

import pytest

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

    def batch_decode(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


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
    _fake_module(
        monkeypatch, "espnet2.bin.s2t_inference_ctc", "Speech2TextGreedySearch", rec
    )

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
    _fake_module(
        monkeypatch, "espnet2.bin.s2t_inference_ctc", "Speech2TextGreedySearch", rec
    )

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
    _fake_module(
        monkeypatch, "espnet2.bin.s2t_inference_ctc", "Speech2TextGreedySearch", rec
    )

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


def test_a_tag_for_another_task_is_explained(monkeypatch, tmp_path, capsys):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")

    class Mismatch:
        @staticmethod
        def from_pretrained(model_tag=None, device=None, **kwargs):
            raise TypeError("__init__() got an unexpected keyword argument 'x'")

    _fake_module(
        monkeypatch,
        "espnet2.bin.s2t_inference_ctc",
        "Speech2TextGreedySearch",
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

    _fake_module(
        monkeypatch, "espnet2.bin.s2t_inference_ctc", "Speech2TextGreedySearch", Bug
    )

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

    _fake_module(
        monkeypatch, "espnet2.bin.s2t_inference_ctc", "Speech2TextGreedySearch", Bug
    )

    with pytest.raises(TypeError, match="'device'"):
        cli.main(["asr", str(audio)])
