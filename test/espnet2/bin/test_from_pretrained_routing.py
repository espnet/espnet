"""Every inference class must fetch a published model through one helper.

The 25 files with a ``from_pretrained`` each carried their own copy of the
espnet_model_zoo boilerplate, and one of the copies is what let a bundle
packed by espnet3 reach an espnet2 constructor and fail there with
``unexpected keyword argument 'training_config'``. These tests call each
``from_pretrained`` with the download and the constructor replaced, so they
assert the routing without fetching anything.
"""

import ast
import pathlib

import pytest

BIN = pathlib.Path(__file__).parents[3] / "espnet2" / "bin"


def _classes_with_from_pretrained():
    for f in sorted(BIN.glob("*.py")):
        src = f.read_text()
        if "download_pretrained(model_tag)" not in src:
            continue
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ClassDef) and any(
                isinstance(x, ast.FunctionDef) and x.name == "from_pretrained"
                for x in node.body
            ):
                yield f.stem, node.name


CLASSES = sorted(_classes_with_from_pretrained())


def test_every_bin_module_with_from_pretrained_was_found():
    # the list is derived, so a new inference class is covered automatically;
    # this guards against the derivation silently finding nothing
    assert len(CLASSES) > 20


@pytest.mark.parametrize("module_name, class_name", CLASSES)
def test_from_pretrained_goes_through_the_shared_helper(
    module_name, class_name, monkeypatch
):
    module = pytest.importorskip(f"espnet2.bin.{module_name}")
    cls = getattr(module, class_name)

    artifacts = {"some_train_config": "c.yaml", "some_model_file": "m.pth"}
    asked = []
    built = {}

    def fake_download(model_tag):
        asked.append(model_tag)
        return dict(artifacts)

    class Recorder:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setattr(module, "download_pretrained", fake_download)
    monkeypatch.setattr(module, class_name, Recorder)

    cls.from_pretrained("espnet/some-model")

    assert asked == ["espnet/some-model"]
    for key, value in artifacts.items():
        assert built[key] == value


@pytest.mark.parametrize("module_name, class_name", CLASSES)
def test_from_pretrained_without_a_tag_downloads_nothing(
    module_name, class_name, monkeypatch
):
    module = pytest.importorskip(f"espnet2.bin.{module_name}")
    cls = getattr(module, class_name)

    def fail(model_tag):  # pragma: no cover - the point is that it is not called
        raise AssertionError("downloaded without a model tag")

    class Recorder:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(module, "download_pretrained", fail)
    monkeypatch.setattr(module, class_name, Recorder)

    cls.from_pretrained(None)
