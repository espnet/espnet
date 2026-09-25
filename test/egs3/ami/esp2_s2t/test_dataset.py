"""Tests for the AMI SOT dataset in egs3/ami/esp2_s2t/dataset/."""

import importlib.util
import os
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
def test_load_utt_ids_covers_the_whole_split_in_cut_order():
    ids = ds_mod.load_utt_ids("test")
    assert len(ids) == 6127
    assert ids[0] == "EN2002a-0.37-11.83"
    assert len(set(ids)) == len(ids)


@ami_sot_paths.needs_corpus
def test_reference_texts_cover_every_utterance_id():
    ids = ds_mod.load_utt_ids("test")
    refs = ds_mod.load_reference_texts("test")
    assert set(refs) == set(ids)


@ami_sot_paths.needs_corpus
def test_dataset_length_matches_the_id_list():
    dataset = ds_mod.AmiSotDataset(split="test")
    assert len(dataset) == len(ds_mod.load_utt_ids("test"))


@ami_sot_paths.needs_corpus
def test_sample_carries_the_four_fields_the_s2t_model_consumes():
    """Exactly these four, no more.

    The dataset pipeline passes the whole dictionary onward, where an unsupported field
    can break a stage, and ESPnetS2TModel.forward takes text_prev and text_ctc as
    required positional arguments.
    """
    dataset = ds_mod.AmiSotDataset(split="test")
    sample = dataset[0]
    assert set(sample) == {"speech", "text", "text_prev", "text_ctc"}
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
    for idx in range(len(dataset)):
        sample = dataset[idx]
        assert set(sample) == {"speech", "text", "text_prev", "text_ctc"}
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
def test_sample_order_matches_load_utt_ids():
    """build_output maps an index back to an id through the built dataset.

    That mapping is only valid while the dataset yields samples in the same order, so
    pin it here.
    """
    ids = ds_mod.load_utt_ids("test")
    refs = ds_mod.load_reference_texts("test")
    dataset = ds_mod.AmiSotDataset(split="test")
    for idx in (0, 1, len(dataset) - 1):
        assert dataset[idx]["text"] == refs[ids[idx]]


class _FakeCut:
    """The part of a cut this module reads, over one second of silence."""

    def __init__(self, cut_id):
        self.id = cut_id
        self.custom = {ds_mod._SOT_TEXT_FIELD: cut_id}

    def load_audio(self):
        return np.zeros((1, 16000), dtype="float32")


def _seed_synthetic_split(monkeypatch, ids):
    """Serve every split from fake cuts, so no test here needs the corpus.

    Replacing the one reader leaves load_utt_ids, load_reference_texts and
    AmiSotDataset running their own code, which is what these tests are about.
    """
    cuts = [_FakeCut(utt) for utt in ids]
    monkeypatch.setattr(ds_mod, "load_cuts", lambda split: cuts)


@pytest.fixture(autouse=True)
def _reset_current_split_record(monkeypatch):
    """Reset current split record.

    Keep get_current_utt_id's split record and cache from leaking between tests:
    constructing an AmiSotDataset writes to real os.environ, which (unlike a plain
    module attribute) survives past the test that set it unless something resets it.
    """
    monkeypatch.delenv(ds_mod._CURRENT_SPLIT_ENV, raising=False)
    monkeypatch.setattr(ds_mod, "_CACHED_SPLIT", None)
    monkeypatch.setattr(ds_mod, "_CACHED_UTT_IDS", None)


def test_constructing_a_dataset_records_its_split(monkeypatch):
    """Constructing a dataset records its split.

    get_current_utt_id must resolve through whichever split an AmiSotDataset
    was actually constructed with, not a hardcoded split name.
    """
    _seed_synthetic_split(monkeypatch, ["x", "y"])
    ds_mod.AmiSotDataset(split="valid")
    assert ds_mod.get_current_utt_id(0) == "x"
    assert os.environ[ds_mod._CURRENT_SPLIT_ENV] == "valid"


def test_a_dataset_that_fails_to_construct_does_not_record_a_split(monkeypatch):
    """A dataset that fails to construct does not record a split.

    A dataset whose split will not read must not record itself: build_output has to
    keep refusing, not silently resolve ids against a split that never loaded.
    """

    def _missing(split):
        raise FileNotFoundError("cuts.jsonl.gz is missing")

    monkeypatch.setattr(ds_mod, "load_cuts", _missing)
    with pytest.raises(FileNotFoundError):
        ds_mod.AmiSotDataset(split="test")
    assert ds_mod._CURRENT_SPLIT_ENV not in os.environ


def test_current_utt_id_raises_before_any_dataset_is_built():
    with pytest.raises(RuntimeError, match="AmiSotDataset"):
        ds_mod.get_current_utt_id(0)


def test_current_utt_id_survives_the_frameworks_own_module_loading(
    monkeypatch, tmp_path
):
    """Current utt id survives the frameworks own module loading.

    The real bug shape: ESPnet3 loads a recipe's own dataset/__init__.py through
    espnet3.components.data.dataset_module's _load_local_dataset_module, under a fresh,
    uniquely named module spec -- not the stable, dotted
    `egs3.ami.esp2_s2t.dataset.dataset` path this test file (and
    src/inference.py) import by.

    A module-level Python global set by one copy would be invisible to the other; only a
    process-wide channel bridges them. This goes through the real framework loader, for
    a split ("valid") that disagrees with the old hardcoded "test", to prove
    get_current_utt_id survives that wall for real, not just when the test happens to
    construct AmiSotDataset the same way inference.py imports it.
    """
    # dataset/config.yaml's split_dirs: train->data/train, valid->data/dev,
    # test->data/test.
    for rel_dir, ids in (("data/test", ["t-0"]), ("data/dev", ["v-0", "v-1"])):
        ami_sot_paths.write_cuts(tmp_path / rel_dir, ids)
    monkeypatch.setenv("AMI_SOT_DATA_ROOT", str(tmp_path))

    from espnet3.components.data.dataset_module import load_dataset_module

    module = load_dataset_module(data_src=None, recipe_dir=str(ami_sot_paths.RECIPE))
    module.Dataset(split="valid")

    assert ds_mod.get_current_utt_id(1) == "v-1"


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
