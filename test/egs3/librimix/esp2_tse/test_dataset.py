"""LibriMix manifests, enrollment modes and raw dataset contracts."""

import json

import numpy as np
import pytest
import soundfile as sf

from egs3.librimix.esp2_tse.dataset import Dataset, DatasetBuilder
from espnet3.components.data.base_dataset import BaseDataset


@pytest.fixture
def corpus(tmp_path):
    uid = "1-1-1_2-2-2"
    for part in ("train", "dev"):
        split = f"2mix_16k_max_{part}_mix-clean"
        manifest = tmp_path / split
        manifest.mkdir()
        root = tmp_path / "LibriMix/libri_mix/Libri2Mix/wav16k/max" / part
        for source in ("mix_clean", "s1", "s2"):
            (root / source).mkdir(parents=True)
            sf.write(root / source / f"{uid}.wav", np.zeros(128), 16000)
        for name, source in (
            ("wav.scp", "mix_clean"),
            ("spk1.scp", "s1"),
            ("spk2.scp", "s2"),
        ):
            (manifest / name).write_text(f"{uid} {root / source / (uid + '.wav')}\n")
        for spk in (1, 2):
            (manifest / f"text_spk{spk}").write_text(f"{uid} text\n")
        if part == "train":
            (manifest / "spk2enroll.json").write_text(
                json.dumps({"1": [["1-1-1", "enroll.wav"]]})
            )
        else:
            for speaker in ("1-1-1", "2-2-2"):
                sf.write(root / "mix_clean" / f"{speaker}.wav", np.zeros(64), 16000)
            (manifest / "mixture2enrollment").write_text(
                f"{uid} 1-1-1 mix_clean/1-1-1\n{uid} 2-2-2 mix_clean/2-2-2\n"
            )
    return tmp_path


def test_raw_fields_and_fixed_enrollment(corpus):
    ds = Dataset(
        split="2mix_16k_max_dev_mix-clean",
        data_dir=corpus,
        ignore_key_prefix=["text_spk", "num_spk"],
    )
    assert isinstance(ds, BaseDataset)
    assert ds[0]["utt_id"] == "1-1-1_2-2-2"
    assert {f.name for f in ds.fields} == set(ds[0])
    assert all(f.kind == "audio" for f in ds.fields if f.name.startswith("enroll"))


def test_training_placeholder_contract(corpus):
    ds = Dataset(
        split="2mix_16k_max_train_mix-clean",
        data_dir=corpus,
        training_placeholders=True,
    )
    assert ds[0]["enroll_ref1"].startswith("*1-1-1 ")
    assert next(f.kind for f in ds.fields if f.name == "enroll_ref1") == "text"
    assert "utt_id" not in ds[0]


def test_missing_manifest_does_not_build_or_download(tmp_path, monkeypatch):
    def unexpected(**kwargs):
        pytest.fail("dataset loading must not build or download")

    monkeypatch.setattr(DatasetBuilder, "build", unexpected)
    with pytest.raises(FileNotFoundError, match="create_dataset|LibriMix"):
        Dataset(split="2mix_16k_max_dev_mix-clean", data_dir=tmp_path)


def test_unknown_split():
    with pytest.raises(ValueError, match="Unknown split"):
        Dataset(split="invalid")


def test_duplicate_manifest_id(corpus):
    path = corpus / "2mix_16k_max_dev_mix-clean/wav.scp"
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        Dataset(split="2mix_16k_max_dev_mix-clean", data_dir=corpus)


def test_missing_enrollment_map_requires_build(corpus):
    (corpus / "2mix_16k_max_dev_mix-clean/mixture2enrollment").unlink()
    with pytest.raises(FileNotFoundError, match="create_dataset"):
        Dataset(split="2mix_16k_max_dev_mix-clean", data_dir=corpus)
