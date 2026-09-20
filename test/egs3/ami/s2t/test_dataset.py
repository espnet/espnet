"""Tests for the AMI SOT dataset in egs3/ami/s2t/dataset/."""

import importlib.util
import os
import sys
from pathlib import Path

import ami_sot_paths
import numpy as np
import pytest

pytest.importorskip("soundfile")


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
def test_load_utt_ids_reads_the_first_column_in_file_order():
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
    # Either shape of prepared corpus is acceptable here: one this recipe
    # built carries the configured prompt, an older one opens on a timestamp.
    prompt = ds_mod._CONFIG.get("prompt", "")
    assert sample["text"].startswith(prompt) or sample["text"].startswith("<|")


@ami_sot_paths.needs_corpus
def test_a_corpus_without_the_optional_files_still_reads():
    """A corpus without the optional files still reads.

    The prepared corpus predates them and cannot be rewritten, so absence must fall back
    rather than raise.
    """
    dataset = ds_mod.AmiSotDataset(split="test")
    sample = dataset[0]
    assert sample["text_prev"] == "<|nospeech|>"
    assert sample["text_ctc"] == "<|nospeech|>"


def test_init_raises_when_the_two_wav_scp_parsers_disagree_in_length(
    monkeypatch,
):
    """Init raises when the two wav scp parsers disagree in length.

    load_utt_ids tolerates an id with no path; _load_wav_paths raises on one today.

    Nothing ties the two parsers together as an invariant, so assert it directly in case
    that ever changes without both being updated together.
    """
    monkeypatch.setattr(ds_mod, "load_utt_ids", lambda split: ["a", "b", "c"])
    monkeypatch.setattr(ds_mod, "_load_wav_paths", lambda split: [Path("x"), Path("y")])
    with pytest.raises(ValueError, match="wav.scp"):
        ds_mod.AmiSotDataset(split="test")


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


def _seed_synthetic_split(monkeypatch, ids):
    """Point load_utt_ids/_load_wav_paths/load_reference_texts at fake data.

    Lets a test build an AmiSotDataset without touching disk or the real corpus, the
    same trick test_init_raises_when_the_two_wav_scp_parsers... already uses for one
    function at a time.
    """
    monkeypatch.setattr(ds_mod, "load_utt_ids", lambda split: list(ids))
    monkeypatch.setattr(
        ds_mod, "_load_wav_paths", lambda split: [Path(utt) for utt in ids]
    )
    monkeypatch.setattr(
        ds_mod,
        "load_reference_texts",
        lambda split, filename="text": {utt: "" for utt in ids},
    )


@pytest.fixture(autouse=True)
def _reset_current_split_record(monkeypatch):
    """Reset current split record.

    Keep current_utt_id's split record and cache from leaking between tests:
    constructing an AmiSotDataset writes to real os.environ, which (unlike a plain
    module attribute) survives past the test that set it unless something resets it.
    """
    monkeypatch.delenv(ds_mod._CURRENT_SPLIT_ENV, raising=False)
    monkeypatch.setattr(ds_mod, "_CACHED_SPLIT", None)
    monkeypatch.setattr(ds_mod, "_CACHED_UTT_IDS", None)


def test_constructing_a_dataset_records_its_split(monkeypatch):
    """Constructing a dataset records its split.

    current_utt_id must resolve through whichever split an AmiSotDataset was actually
    constructed with, not a hardcoded split name.
    """
    _seed_synthetic_split(monkeypatch, ["x", "y"])
    ds_mod.AmiSotDataset(split="valid")
    assert ds_mod.current_utt_id(0) == "x"
    assert os.environ[ds_mod._CURRENT_SPLIT_ENV] == "valid"


def test_a_dataset_that_fails_to_construct_does_not_record_a_split(monkeypatch):
    """A dataset that fails to construct does not record a split.

    A half-built dataset (the wav.scp parser disagreement below) must not record itself:
    build_output must keep refusing, not silently resolve ids against a split that never
    finished loading.
    """
    monkeypatch.setattr(ds_mod, "load_utt_ids", lambda split: ["a", "b", "c"])
    monkeypatch.setattr(ds_mod, "_load_wav_paths", lambda split: [Path("x")])
    with pytest.raises(ValueError, match="wav.scp"):
        ds_mod.AmiSotDataset(split="test")
    assert ds_mod._CURRENT_SPLIT_ENV not in os.environ


def test_current_utt_id_raises_before_any_dataset_is_built():
    with pytest.raises(RuntimeError, match="AmiSotDataset"):
        ds_mod.current_utt_id(0)


def test_current_utt_id_survives_the_frameworks_own_module_loading(
    monkeypatch, tmp_path
):
    """Current utt id survives the frameworks own module loading.

    The real bug shape: ESPnet3 loads a recipe's own dataset/__init__.py through
    espnet3.components.data.dataset_module's _load_local_dataset_module, under a fresh,
    uniquely named module spec -- not the stable, dotted `egs3.ami.s2t.dataset.dataset`
    path this test file (and src/inference.py) import by.

    A module-level Python global set by one copy would be invisible to the other; only a
    process-wide channel bridges them. This goes through the real framework loader, for
    a split ("valid") that disagrees with the old hardcoded "test", to prove
    current_utt_id survives that wall for real, not just when the test happens to
    construct AmiSotDataset the same way inference.py imports it.
    """
    # dataset/config.yaml's split_dirs: train->data/train, valid->data/dev,
    # test->data/test.
    for split, rel_dir, ids in (
        ("test", "data/test", ["t-0"]),
        ("valid", "data/dev", ["v-0", "v-1"]),
    ):
        split_dir = tmp_path / rel_dir
        split_dir.mkdir(parents=True)
        (split_dir / "wav.scp").write_text(
            "".join(f"{utt} {utt}.wav\n" for utt in ids), encoding="utf-8"
        )
        (split_dir / "text").write_text(
            "".join(f"{utt}\n" for utt in ids), encoding="utf-8"
        )
    monkeypatch.setenv("AMI_SOT_DATA_ROOT", str(tmp_path))

    from espnet3.components.data.dataset_module import load_dataset_module

    module = load_dataset_module(data_src=None, recipe_dir=str(ami_sot_paths.RECIPE))
    module.Dataset(split="valid")

    assert ds_mod.current_utt_id(1) == "v-1"


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
    monkeypatch.setitem(builder_mod._CONFIG, "data_root", str(tmp_path))
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

    package = importlib.import_module("egs3.ami.s2t.dataset")
    from egs3.ami.s2t.dataset.builder import AmiSotBuilder
    from egs3.ami.s2t.dataset.dataset import AmiSotDataset

    assert package.Dataset is AmiSotDataset
    assert package.DatasetBuilder is AmiSotBuilder
