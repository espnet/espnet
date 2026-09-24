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


def _from_pretrained_methods():
    """Yield (module, class, argument names) for every from_pretrained."""
    for f in sorted(BIN.glob("*.py")):
        for node in ast.walk(ast.parse(f.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            for x in node.body:
                if isinstance(x, ast.FunctionDef) and x.name == "from_pretrained":
                    args = [a.arg for a in x.args.args + x.args.kwonlyargs]
                    yield f.stem, node.name, args


# Membership is decided by the signature, not by what the file already
# contains: a class that took a model_tag and fetched it its own way would
# be tested like the rest, and fail.
CLASSES = sorted(
    (module, name)
    for module, name, args in _from_pretrained_methods()
    if "model_tag" in args
)
NO_MODEL_TAG = sorted(
    (module, name)
    for module, name, args in _from_pretrained_methods()
    if "model_tag" not in args
)


def test_every_inference_class_that_takes_a_model_tag_was_found():
    # the list is derived, so a new inference class is covered without
    # touching this file; this guards against the derivation finding nothing
    assert len(CLASSES) > 20


def test_only_s2st_offers_from_pretrained_without_a_model_tag():
    # Speech2Speech.from_pretrained takes only vocoder_tag, so an S2ST model
    # cannot be loaded from a published tag at all. If that is ever fixed,
    # the class joins CLASSES above and this expectation goes away.
    assert NO_MODEL_TAG == [("s2st_inference", "Speech2Speech")]


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

    def record(self, **kwargs):
        built.update(kwargs)

    monkeypatch.setattr(module, "download_pretrained", fake_download)
    # The constructor is replaced on the class rather than the module, so
    # that a from_pretrained written as a classmethod - which calls
    # cls(...), never the module-level name - is intercepted too.
    monkeypatch.setattr(cls, "__init__", record)

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

    monkeypatch.setattr(module, "download_pretrained", fail)
    monkeypatch.setattr(cls, "__init__", lambda self, **kwargs: None)

    cls.from_pretrained(None)
