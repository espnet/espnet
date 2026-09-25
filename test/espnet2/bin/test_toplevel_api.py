"""The top-level ``espnet`` package: its version string and ``espnet.load``.

These cover ``espnet/__init__.py`` rather than anything under ``espnet2/bin``,
but they live here because this is where CI runs them: ``ci/test_python_*.sh``
run ``pytest test/espnet2`` and ``pytest test/espnet3/`` and nothing else, so a
``test/espnet/`` directory would be collected by no job at all. ``espnet.load``
is a front end onto the ``espnet2.bin.*_inference`` classes, which makes
``test/espnet2/bin`` its natural home in any case.

No model is downloaded. Each test replaces the inference class that ``load``
imports, so what is checked is the wiring: which class is asked for which tag,
what it is called with, and what is said when the answer is unclear.
"""

import importlib
import subprocess
import sys
import types

import pytest

import espnet


def _fake_class(monkeypatch, task, recorder):
    """Put ``recorder`` where ``load`` looks for the class serving ``task``."""
    module_name, class_name = espnet.TASKS[task]
    module = types.ModuleType(module_name)
    setattr(module, class_name, recorder)
    monkeypatch.setitem(sys.modules, module_name, module)
    return recorder


class _Recorder:
    """Stands in for an inference class: records how it was built."""

    __name__ = "Recorder"

    def __init__(self):
        self.tag = None
        self.device = None
        self.kwargs = None

    def from_pretrained(self, model_tag=None, device=None, **kwargs):
        self.tag, self.device, self.kwargs = model_tag, device, kwargs
        return self


def test_version_is_a_string():
    assert isinstance(espnet.__version__, str)
    assert espnet.__version__


def test_a_source_tree_that_was_never_installed_still_has_a_version():
    # the fallback only runs at import time, so the module is re-executed with
    # the metadata lookup failing the way it does in an uninstalled checkout
    import importlib.metadata

    real = importlib.metadata.version

    def not_installed(name):
        raise importlib.metadata.PackageNotFoundError(name)

    importlib.metadata.version = not_installed
    try:
        importlib.reload(espnet)
        assert isinstance(espnet.__version__, str)
        assert "not installed" in espnet.__version__
    finally:
        importlib.metadata.version = real
        importlib.reload(espnet)

    assert isinstance(espnet.__version__, str)


def test_hub_labels_merge_the_pipeline_tag_with_the_tags(monkeypatch):
    # one repository's labels arrive in two places and either may be empty;
    # the task table is matched against both at once
    class Info:
        pipeline_tag = "audio-classification"
        tags = ["espnet", "speaker-verification"]

    monkeypatch.setattr("huggingface_hub.model_info", lambda tag: Info())

    assert espnet._hub_labels("espnet/some-model") == {
        "audio-classification",
        "espnet",
        "speaker-verification",
    }


def test_a_repository_that_carries_no_labels_at_all(monkeypatch):
    class Bare:
        pipeline_tag = None
        tags = None

    monkeypatch.setattr("huggingface_hub.model_info", lambda tag: Bare())

    assert espnet._hub_labels("espnet/some-model") == set()


def test_importing_espnet_does_not_drag_in_torch():
    # the promise of a light top level: `import espnet` must stay cheap, so
    # the module body may not import torch or espnet2 (`load` does, on call)
    # a fresh interpreter, because pytest has already imported both here
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, espnet; "
            "print(sorted(m for m in sys.modules "
            "if m == 'torch' or m.startswith('espnet2')))",
        ],
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "[]", probe.stdout


@pytest.mark.parametrize("task", sorted(espnet.TASKS))
def test_each_task_dispatches_to_its_own_class(task, monkeypatch):
    rec = _fake_class(monkeypatch, task, _Recorder())

    built = espnet.load("espnet/some-model", task=task)

    assert built is rec
    assert rec.tag == "espnet/some-model"
    assert rec.device == "cpu"


def test_every_task_names_a_class_that_exists():
    # a typo in TASKS would only show as an ImportError on a user's machine.
    # Imported outright rather than skipped when missing: every one of these
    # is an inference entry point that ci/check_inference_imports.py already
    # requires to import on a bare install, so a skip here would only hide
    # the typo this test exists to catch.
    for module_name, class_name in espnet.TASKS.values():
        module = importlib.import_module(module_name)
        assert hasattr(module, class_name), (module_name, class_name)


def test_device_and_extra_arguments_reach_the_constructor(monkeypatch):
    rec = _fake_class(monkeypatch, "asr", _Recorder())

    espnet.load("espnet/some-asr", task="asr", device="cuda:1", beam_size=1)

    assert rec.device == "cuda:1"
    assert rec.kwargs == {"beam_size": 1}


def test_an_explicit_task_wins_and_looks_nothing_up(monkeypatch):
    def explode(model_tag):  # pragma: no cover - the point is that it is unused
        raise AssertionError("asked the Hub about a tag whose task was given")

    monkeypatch.setattr(espnet, "_hub_labels", explode)
    rec = _fake_class(monkeypatch, "tts", _Recorder())

    # the Hub calls this repository a speaker model; the caller knows better
    assert espnet.load("espnet/mislabelled", task="tts") is rec


def test_an_unknown_task_is_refused_by_name(monkeypatch):
    def explode(model_tag):  # pragma: no cover - refused before any lookup
        raise AssertionError("asked the Hub about a tag with an invalid task")

    monkeypatch.setattr(espnet, "_hub_labels", explode)

    with pytest.raises(ValueError) as e:
        espnet.load("espnet/some-model", task="speech2speech")

    message = str(e.value)
    assert "speech2speech" in message
    for task in espnet.TASKS:
        assert task in message


@pytest.mark.parametrize(
    "labels, task",
    [
        ({"text-to-speech"}, "tts"),
        ({"text-to-audio"}, "tts"),
        ({"audio-to-audio"}, "enh"),
        ({"voice-activity-detection"}, "diar"),
        # espnet publishes its speaker models as audio-classification, which
        # on its own names no task; the speaker tag beside it does
        ({"audio-classification", "speaker-verification"}, "spk"),
        ({"audio-classification", "speaker-recognition"}, "spk"),
    ],
)
def test_the_task_is_inferred_from_the_hub_labels(labels, task, monkeypatch):
    monkeypatch.setattr(espnet, "_hub_labels", lambda tag: labels)
    rec = _fake_class(monkeypatch, task, _Recorder())

    assert espnet.load("espnet/some-model") is rec


@pytest.mark.parametrize(
    "keys, task",
    [
        ({"yaml_files": {"s2t_train_config": "c"}}, "s2t"),
        ({"yaml_files": {"asr_train_config": "c"}}, "asr"),
        ({"files": {"s2t_model_file": "m"}}, "s2t"),
    ],
)
def test_asr_and_s2t_are_split_by_the_packed_meta_yaml(
    keys, task, monkeypatch, tmp_path
):
    # both are automatic-speech-recognition on the Hub, and the two tasks
    # now name classes of the same name in different modules, so the labels
    # alone cannot choose asr_inference from s2t_inference
    import yaml

    meta = tmp_path / "meta.yaml"
    meta.write_text(yaml.safe_dump(keys), encoding="utf-8")
    monkeypatch.setattr(
        espnet, "_hub_labels", lambda tag: {"automatic-speech-recognition"}
    )
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: str(meta))
    rec = _fake_class(monkeypatch, task, _Recorder())

    assert espnet.load("espnet/some-model") is rec


def test_a_model_the_hub_says_nothing_about_asks_for_the_task(monkeypatch):
    monkeypatch.setattr(espnet, "_hub_labels", lambda tag: set())

    with pytest.raises(ValueError) as e:
        espnet.load("/some/local/directory")

    message = str(e.value)
    assert "/some/local/directory" in message
    for task in espnet.TASKS:
        assert task in message


def test_a_tag_for_another_task_is_explained(monkeypatch):
    from espnet2.utils.pretrained import ModelTagError

    class Mismatch:
        __name__ = "Text2Speech"

        def __init__(self, train_config=None, model_file=None, device="cpu"):
            pass

        @staticmethod
        def from_pretrained(model_tag=None, device=None, **kwargs):
            raise TypeError(
                "__init__() got an unexpected keyword argument 'asr_train_config'"
            )

    _fake_class(monkeypatch, "tts", Mismatch)

    with pytest.raises(ModelTagError) as e:
        espnet.load("espnet/an-asr-model", task="tts")

    message = str(e.value)
    assert "does not look like a model for espnet.load(task='tts')" in message
    assert "asr_train_config" in message


def test_an_argument_the_caller_invented_is_not_blamed_on_the_model(monkeypatch):
    from espnet2.utils.pretrained import ModelTagError

    class Strict:
        __name__ = "Text2Speech"

        def __init__(self, train_config=None, model_file=None, device="cpu"):
            pass

        @staticmethod
        def from_pretrained(model_tag=None, device=None, **kwargs):
            raise TypeError("__init__() got an unexpected keyword argument 'beam_size'")

    _fake_class(monkeypatch, "tts", Strict)

    # from_pretrained merges the caller's arguments with the model's own, so
    # an unexpected keyword is only the model's fault when the caller did not
    # pass it; here they did, and the real TypeError has to stand
    with pytest.raises(TypeError) as e:
        espnet.load("espnet/a-tts-model", task="tts", beam_size=1)

    assert not isinstance(e.value, ModelTagError)
    assert "beam_size" in str(e.value)


def test_a_tag_with_no_hub_repository_carries_no_labels():
    # a local directory or a URL is a legitimate model_tag for the downloader
    # but names no repository, so there is nothing to read and nothing to fail
    assert espnet._hub_labels("/no/such/directory") == set()


@pytest.mark.parametrize("meta", ["not a mapping", {"files": {}}])
def test_an_unreadable_meta_yaml_leaves_asr_and_s2t_undecided(
    meta, monkeypatch, tmp_path
):
    import yaml

    path = tmp_path / "meta.yaml"
    path.write_text(yaml.safe_dump(meta), encoding="utf-8")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: str(path))

    # a mapping with no artifact keys still answers "asr"; only a file that is
    # not a mapping at all leaves the question open
    expected = None if meta == "not a mapping" else "asr"
    assert espnet._asr_or_s2t("espnet/some-model") == expected


def test_a_meta_yaml_that_cannot_be_fetched_asks_for_the_task(monkeypatch):
    def missing(*args, **kwargs):
        raise OSError("meta.yaml is not in this repository")

    monkeypatch.setattr(
        espnet, "_hub_labels", lambda tag: {"automatic-speech-recognition"}
    )
    monkeypatch.setattr("huggingface_hub.hf_hub_download", missing)

    with pytest.raises(ValueError) as e:
        espnet.load("espnet/hand-uploaded")

    assert "espnet/hand-uploaded" in str(e.value)
