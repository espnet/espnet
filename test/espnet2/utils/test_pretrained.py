import sys

import pytest

from espnet2.utils.pretrained import download_pretrained


def _fake_downloader(monkeypatch, artifacts):
    module = type(sys)("espnet_model_zoo.downloader")

    class ModelDownloader:
        def download_and_unpack(self, model_tag):
            return dict(artifacts)

    module.ModelDownloader = ModelDownloader
    monkeypatch.setitem(sys.modules, "espnet_model_zoo", type(sys)("espnet_model_zoo"))
    monkeypatch.setitem(sys.modules, "espnet_model_zoo.downloader", module)


def test_espnet2_pack_returns_the_constructor_kwargs(monkeypatch):
    artifacts = {"asr_train_config": "c.yaml", "asr_model_file": "m.pth"}
    _fake_downloader(monkeypatch, artifacts)
    assert download_pretrained("espnet/some_asr") == artifacts


@pytest.mark.parametrize("key", ["inference_config", "training_config"])
def test_an_espnet3_pack_says_which_loader_to_use(monkeypatch, key):
    _fake_downloader(monkeypatch, {key: "conf/x.yaml"})
    with pytest.raises(RuntimeError) as e:
        download_pretrained("espnet/some_espnet3_model")
    assert "espnet3" in str(e.value)
    assert "InferenceModel.from_pretrained" in str(e.value)


def test_a_missing_model_zoo_is_named(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "espnet_model_zoo", None)
    monkeypatch.setitem(sys.modules, "espnet_model_zoo.downloader", None)
    with pytest.raises(ImportError):
        download_pretrained("espnet/anything")
    assert "espnet_model_zoo" in caplog.text
