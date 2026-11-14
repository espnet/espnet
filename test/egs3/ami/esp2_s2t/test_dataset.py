"""Tests for the AMI SOT dataset in egs3/ami/esp2_s2t/dataset/."""

import importlib.util
import sys

import ami_sot_paths
import numpy as np
import pytest

pytest.importorskip("lhotse")


def _load_dataset_module():
    sys.path.insert(0, str(ami_sot_paths.REPO))
    spec = importlib.util.spec_from_file_location(
        "ami_s2t_dataset", ami_sot_paths.RECIPE / "dataset" / "dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ds_mod = _load_dataset_module()


@ami_sot_paths.needs_corpus
def test_a_sample_carries_its_own_cut_id():
    """Inference writes every output under this id, so it is the cut's own."""
    cuts = ds_mod.load_cuts("test")
    ids = [c.id for c in cuts]
    assert len(ids) == 6127
    assert len(set(ids)) == len(ids)
    dataset = ds_mod.AmiSotDataset(split="test")
    assert len(dataset) == len(ids)
    assert dataset[0]["utt_id"] == ids[0] == "EN2002a-0.37-11.83"


@ami_sot_paths.needs_corpus
def test_sample_carries_the_four_model_fields_and_its_id():
    """Exactly these five, no more.

    ESPnetS2TModel.forward takes text_prev and text_ctc as required positional
    arguments. utt_id is the one field the model does not take; the
    preprocessor drops it before training.
    """
    dataset = ds_mod.AmiSotDataset(split="test")
    sample = dataset[0]
    assert set(sample) == {"utt_id", "speech", "text", "text_prev", "text_ctc"}
    assert isinstance(sample["speech"], np.ndarray)
    assert sample["speech"].dtype == np.float32
    assert sample["speech"].ndim == 1
    prompt = ds_mod._CONFIG["prompt"]
    assert sample["text"].startswith(prompt)


@ami_sot_paths.needs_corpus
def test_a_sample_reports_no_previous_text_and_no_ctc_target():
    """Both text fields carry the not-available symbol.

    This recipe conditions on no previous text and trains with ctc_weight 0.0, so
    neither field ever carries content; a sample that grew one would change what the
    model is trained on.
    """
    dataset = ds_mod.AmiSotDataset(split="test")
    sample = dataset[0]
    assert sample["text_prev"] == "<|nospeech|>"
    assert sample["text_ctc"] == "<|nospeech|>"


def test_every_sample_reports_no_previous_text_and_no_ctc_target(monkeypatch):
    """The two fields the model takes but this recipe never fills.

    ESPnetS2TModel.forward takes text_prev and text_ctc as required positional
    arguments. This recipe conditions on no previous text and trains with
    ctc_weight 0.0, so both carry the not-available symbol on every sample; one
    that grew content would change what the model is trained on. Its sibling
    above reads the real corpus, which most machines do not have, so this one
    runs off fake cuts and covers every sample rather than the first.
    """
    _seed_synthetic_split(monkeypatch, ["x", "y", "z"])
    dataset = ds_mod.AmiSotDataset(split="test")

    na = ds_mod._CONFIG["na_symbol"]
    assert len(dataset) == 3
    # The first item is also checked against AmiSotDataset.fields here.
    for idx in range(len(dataset)):
        sample = dataset[idx]
        assert set(sample) == {"utt_id", "speech", "text", "text_prev", "text_ctc"}
        assert sample["utt_id"] == "xyz"[idx]
        assert sample["text_prev"] == na
        assert sample["text_ctc"] == na


def test_reading_an_unbuilt_split_names_the_file_to_produce(tmp_path, monkeypatch):
    """The error has to name the missing file and the stage that writes it.

    A split is one file now, so this is the only way reading one can fail, and the
    message is all a user has to go on.
    """
    # setenv, not setitem: data_root interpolates ${oc.env:AMI_SOT_DATA_ROOT}
    # lazily, and assigning over it freezes the node at its current value for
    # the rest of the session -- monkeypatch's undo restores the resolved
    # string, not the interpolation, so every later test in this process would
    # read whatever root this one happened to use.
    monkeypatch.setenv("AMI_SOT_DATA_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="create_dataset"):
        ds_mod.load_cuts("test")


@ami_sot_paths.needs_corpus
def test_a_sample_is_its_cut_id_and_its_cut_text():
    """The id and the reference come from the same cut, so they cannot drift."""
    cuts = ds_mod.load_cuts("test")
    dataset = ds_mod.AmiSotDataset(split="test")
    for idx in (0, 1, len(dataset) - 1):
        sample = dataset[idx]
        assert sample["utt_id"] == cuts[idx].id
        assert sample["text"] == cuts[idx].custom[ds_mod._SOT_TEXT_FIELD]


class _FakeCut:
    """The part of a cut this module reads, over one second of silence."""

    def __init__(self, cut_id, sampling_rate=16000, channels=1):
        self.id = cut_id
        self.custom = {ds_mod._SOT_TEXT_FIELD: cut_id}
        self.sampling_rate = sampling_rate
        self._channels = channels

    def load_audio(self):
        return np.zeros((self._channels, self.sampling_rate), dtype="float32")


def _seed_synthetic_split(monkeypatch, ids, **cut_args):
    """Serve every split from fake cuts, so no test here needs the corpus."""
    cuts = [_FakeCut(utt, **cut_args) for utt in ids]
    monkeypatch.setattr(ds_mod, "load_cuts", lambda split: cuts)


def test_building_another_split_leaves_the_first_ones_ids_alone(monkeypatch):
    """Each dataset keeps its own cuts, so its ids are its own.

    infer builds every test set in one process, and measure builds them again
    to read the references; a later split must not rename an earlier one.
    """
    splits = {"valid": [_FakeCut("v-0"), _FakeCut("v-1")], "test": [_FakeCut("t-0")]}
    monkeypatch.setattr(ds_mod, "load_cuts", lambda split: splits[split])
    valid = ds_mod.AmiSotDataset(split="valid")
    test = ds_mod.AmiSotDataset(split="test")

    assert [valid[i]["utt_id"] for i in range(len(valid))] == ["v-0", "v-1"]
    assert test[0]["utt_id"] == "t-0"


@pytest.mark.parametrize("cut_args", [{"sampling_rate": 8000}, {"channels": 2}])
def test_audio_that_is_not_16k_mono_is_refused(monkeypatch, cut_args):
    """A bare array cannot say its rate, so the dataset says it for it.

    The training preprocessor and the inference contract both read the array as
    16 kHz mono; anything else would be decoded at the wrong speed or with a
    channel silently dropped.
    """
    _seed_synthetic_split(monkeypatch, ["x"], **cut_args)
    with pytest.raises(ValueError, match="16 kHz mono"):
        ds_mod.AmiSotDataset(split="test")[0]


def _load_builder_module():
    sys.path.insert(0, str(ami_sot_paths.REPO))
    spec = importlib.util.spec_from_file_location(
        "ami_s2t_builder", ami_sot_paths.RECIPE / "dataset" / "builder.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@ami_sot_paths.needs_corpus
def test_builder_reports_a_prepared_corpus_as_built():
    """Builder reports a prepared corpus as built.

    is_built inspects the split directories, not the build inputs, so a corpus prepared
    by any means counts as built.
    """
    builder = _load_builder_module().AmiSotBuilder()
    assert builder.is_built() is True


def test_builder_reports_an_empty_root_as_neither_built_nor_prepared(
    tmp_path, monkeypatch
):
    builder_mod = _load_builder_module()
    monkeypatch.setenv("AMI_SOT_DATA_ROOT", str(tmp_path))
    builder = builder_mod.AmiSotBuilder()
    assert builder.is_built() is False
    assert builder.is_source_prepared() is False


def test_dataset_package_exports_the_names_the_framework_looks_up():
    """Dataset package exports the names the framework looks up.

    BaseSystem resolves Dataset/DatasetBuilder via getattr(module, name) on
    dataset/__init__.py (loaded by literal module name), not by loading dataset.py or
    builder.py directly.

    Every other test in this file loads those two files by path and bypasses __init__.py
    entirely, so renaming AmiSotDataset or AmiSotBuilder without updating the re-export
    would break the infer stage while the rest of this suite kept passing.
    """
    import importlib

    package = importlib.import_module("egs3.ami.esp2_s2t.dataset")
    from egs3.ami.esp2_s2t.dataset.builder import AmiSotBuilder
    from egs3.ami.esp2_s2t.dataset.dataset import AmiSotDataset

    assert package.Dataset is AmiSotDataset
    assert package.DatasetBuilder is AmiSotBuilder
