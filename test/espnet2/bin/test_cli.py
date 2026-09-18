"""The `espnet` command: dispatch, defaults and the errors a user can fix.

No model is downloaded here. Each test replaces the inference class the
subcommand imports, so what is checked is the wiring: which class is asked
for which tag, what it is called with, and what reaches the terminal.
"""

import sys
import types

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
