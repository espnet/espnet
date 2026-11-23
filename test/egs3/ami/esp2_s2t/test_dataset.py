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
    """Each sample has the id of its own cut, and the 6127 test ids are unique.

    Inference writes every output under this id.
    """
    cuts = ds_mod.load_cuts("test")
    ids = [c.id for c in cuts]
    assert len(ids) == 6127
    assert len(set(ids)) == len(ids)
    dataset = ds_mod.AmiSotDataset(split="test")
    assert len(dataset) == len(ids)
    assert dataset[0]["utt_id"] == ids[0] == "EN2002a-0.37-11.83"


@ami_sot_paths.needs_corpus
def test_sample_carries_the_four_model_fields_and_its_id():
    """A sample has exactly five fields, the four model inputs and utt_id.

    ESPnetS2TModel.forward takes text_prev and text_ctc as required positional
    arguments. The model does not take utt_id, and the preprocessor drops it
    before training.
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
    """text_prev and text_ctc both contain the not-available symbol.

    This recipe does not condition on previous text and trains with ctc_weight 0.0,
    so neither field has content. Content in either field would change what the
    model is trained on.
    """
    dataset = ds_mod.AmiSotDataset(split="test")
    sample = dataset[0]
    assert sample["text_prev"] == "<|nospeech|>"
    assert sample["text_ctc"] == "<|nospeech|>"


def test_every_sample_reports_no_previous_text_and_no_ctc_target(monkeypatch):
    """Every sample has the not-available symbol in text_prev and text_ctc.

    ESPnetS2TModel.forward takes both fields as required positional arguments. This
    recipe does not condition on previous text and trains with ctc_weight 0.0, so
    content in either field would change what the model is trained on. The test
    above checks the first sample of the real corpus, which most machines do not
    have. This test uses fake cuts and checks every sample.
    """
    _seed_synthetic_split(monkeypatch, ["x", "y", "z"])
    dataset = ds_mod.AmiSotDataset(split="test")

    na = ds_mod._CONFIG["na_symbol"]
    assert len(dataset) == 3
    # BaseDataset also checks the first item against AmiSotDataset.fields.
    for idx in range(len(dataset)):
        sample = dataset[idx]
        assert set(sample) == {"utt_id", "speech", "text", "text_prev", "text_ctc"}
        assert sample["utt_id"] == "xyz"[idx]
        assert sample["text_prev"] == na
        assert sample["text_ctc"] == na


def test_reading_an_unbuilt_split_names_the_file_to_produce(tmp_path, monkeypatch):
    """load_cuts raises FileNotFoundError for a split that is not built.

    The message gives the missing file and create_dataset, the stage that writes
    it. A split is now one file, so this is the only way that reading a split can
    fail. The message is the only help that the user gets.
    """
    # The test sets the environment variable with setenv. data_root interpolates
    # ${oc.env:AMI_SOT_DATA_ROOT} lazily, so assigning a value with setitem would
    # freeze the node at its current value for the rest of the session.
    # monkeypatch's undo would restore the resolved string and lose the
    # interpolation. Every later test in this process would then read this test's
    # root.
    monkeypatch.setenv("AMI_SOT_DATA_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="create_dataset"):
        ds_mod.load_cuts("test")


@ami_sot_paths.needs_corpus
def test_a_sample_is_its_cut_id_and_its_cut_text():
    """A sample's utt_id and text come from the same cut.

    So the id and the reference text cannot get out of sync.
    """
    cuts = ds_mod.load_cuts("test")
    dataset = ds_mod.AmiSotDataset(split="test")
    for idx in (0, 1, len(dataset) - 1):
        sample = dataset[idx]
        assert sample["utt_id"] == cuts[idx].id
        assert sample["text"] == cuts[idx].custom[ds_mod._SOT_TEXT_FIELD]


class _FakeCut:
    """A fake cut with only the parts of a cut that this module reads.

    Its audio is one second of silence.
    """

    def __init__(self, cut_id, sampling_rate=16000, channels=1):
        self.id = cut_id
        self.custom = {ds_mod._SOT_TEXT_FIELD: cut_id}
        self.sampling_rate = sampling_rate
        self._channels = channels

    def load_audio(self):
        return np.zeros((self._channels, self.sampling_rate), dtype="float32")


def _seed_synthetic_split(monkeypatch, ids, **cut_args):
    """Make load_cuts return fake cuts for every split, so no test needs the corpus."""
    cuts = [_FakeCut(utt, **cut_args) for utt in ids]
    monkeypatch.setattr(ds_mod, "load_cuts", lambda split: cuts)


def test_building_another_split_leaves_the_first_ones_ids_alone(monkeypatch):
    """Building a second split does not change the ids of the first split.

    infer builds every test set in one process, and measure builds them again to
    read the references. Each dataset keeps its own cuts, so a later split cannot
    rename the samples of an earlier one.
    """
    splits = {"valid": [_FakeCut("v-0"), _FakeCut("v-1")], "test": [_FakeCut("t-0")]}
    monkeypatch.setattr(ds_mod, "load_cuts", lambda split: splits[split])
    valid = ds_mod.AmiSotDataset(split="valid")
    test = ds_mod.AmiSotDataset(split="test")

    assert [valid[i]["utt_id"] for i in range(len(valid))] == ["v-0", "v-1"]
    assert test[0]["utt_id"] == "t-0"


@pytest.mark.parametrize("cut_args", [{"sampling_rate": 8000}, {"channels": 2}])
def test_audio_that_is_not_16k_mono_is_refused(monkeypatch, cut_args):
    """The dataset raises ValueError for audio that is not 16 kHz mono.

    A plain array has no sampling rate, so the dataset checks the rate itself. The
    training preprocessor and the inference contract both read the array as 16 kHz
    mono. Other audio would be decoded at the wrong speed, or a channel would be
    dropped without an error.
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
    """AmiSotBuilder.is_built() returns True for a prepared corpus.

    is_built only checks the split directories and ignores the build inputs. So a
    corpus prepared in any way counts as built.
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
    """The dataset package exports Dataset and DatasetBuilder for the framework.

    BaseSystem finds Dataset and DatasetBuilder with getattr(module, name) on
    dataset/__init__.py, which it loads by its literal module name. It does not load
    dataset.py or builder.py directly.

    Every other test in this file loads those two files by path and skips
    __init__.py. If AmiSotDataset or AmiSotBuilder were renamed without an update to
    the re-export, the infer stage would break, and the other tests would still pass.
    """
    import importlib

    package = importlib.import_module("egs3.ami.esp2_s2t.dataset")
    from egs3.ami.esp2_s2t.dataset.builder import AmiSotBuilder
    from egs3.ami.esp2_s2t.dataset.dataset import AmiSotDataset

    assert package.Dataset is AmiSotDataset
    assert package.DatasetBuilder is AmiSotBuilder
