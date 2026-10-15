import sys
import types
from test.espnet3.systems.esp2_gan_tts._gan_dummies import DummyDataset
from test.espnet3.systems.esp2_gan_tts._xvector_helpers import (
    write_manifest,
    write_wav,
)

import numpy as np
import pytest

from espnet3.components.data import data_organizer as data_organizer_module
from espnet3.systems.esp2_gan_tts.xvector_provider import XVectorProvider
from espnet3.systems.esp2_gan_tts.xvector_runner import XVectorRunner


@pytest.fixture
def patch_dataset_reference(monkeypatch):
    """Keep DataOrganizer off the filesystem when a module builds datasets."""
    monkeypatch.setattr(
        data_organizer_module,
        "instantiate_dataset_reference",
        lambda config, recipe_dir=None: DummyDataset(),
    )


@pytest.fixture
def manifest(tmp_path):
    """A two-speaker, three-utterance manifest with real wav files."""
    rows = []
    for utt_id, spk in (("u1", "0"), ("u2", "0"), ("u3", "1")):
        wav_path = write_wav(tmp_path, f"{utt_id}.wav")
        rows.append(f"{utt_id}\t{wav_path}\thello\t{spk}\n")
    return write_manifest(tmp_path, rows)


@pytest.fixture
def stub_model(monkeypatch):
    """Replace the network-bound model build and the extractor."""
    monkeypatch.setattr(
        XVectorProvider, "_build_model", staticmethod(lambda *a, **k: "MODEL")
    )
    monkeypatch.setattr(
        XVectorRunner,
        "_extract_embedding",
        staticmethod(
            lambda wav, sr, model, toolkit, device: np.zeros(192, dtype=np.float32)
        ),
    )


@pytest.fixture
def fake_speechbrain(monkeypatch):
    """Install a stub ``speechbrain`` so its code paths run without the dep.

    ``speechbrain`` is an optional extra, so without this the two branches
    that import it would only ever be skipped, never covered.
    """
    for name in (
        "speechbrain",
        "speechbrain.dataio",
        "speechbrain.dataio.preprocess",
        "speechbrain.inference",
        "speechbrain.inference.classifiers",
    ):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    preprocess = sys.modules["speechbrain.dataio.preprocess"]
    if not hasattr(preprocess, "AudioNormalizer"):

        class _AudioNormalizer:
            def __call__(self, wav, sample_rate):
                return wav

        monkeypatch.setattr(
            preprocess, "AudioNormalizer", _AudioNormalizer, raising=False
        )
    return sys.modules["speechbrain.inference.classifiers"]
