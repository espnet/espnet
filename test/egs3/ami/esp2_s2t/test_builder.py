"""Tests for the AMI SOT builder in egs3/ami/esp2_s2t/dataset/builder.py."""

import importlib.util
import sys
from types import SimpleNamespace

import ami_sot_paths
import pytest

pytest.importorskip("soundfile")
pytest.importorskip("lhotse")


def _load_builder_module():
    sys.path.insert(0, str(ami_sot_paths.REPO))
    spec = importlib.util.spec_from_file_location(
        "ami_s2t_builder", ami_sot_paths.RECIPE / "dataset" / "builder.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bl = _load_builder_module()


def test_relocate_remaps_a_recording_whose_stored_path_is_gone(tmp_path):
    source = SimpleNamespace(type="file", source="/gone/EN2002a.Array1-01.wav")
    cut = SimpleNamespace(recording=SimpleNamespace(sources=[source]))
    bl._relocate_recording(cut, tmp_path)
    assert source.source == str(
        tmp_path / "wav_db" / "EN2002a" / "audio" / "EN2002a.Array1-01.wav"
    )


def test_relocate_leaves_a_recording_that_still_resolves(tmp_path):
    """A corpus prepared on this machine must keep working untouched."""
    present = tmp_path / "here.wav"
    present.write_bytes(b"")
    source = SimpleNamespace(type="file", source=str(present))
    cut = SimpleNamespace(recording=SimpleNamespace(sources=[source]))
    bl._relocate_recording(cut, tmp_path / "elsewhere")
    assert source.source == str(present)


def test_relocate_ignores_non_file_sources(tmp_path):
    source = SimpleNamespace(type="command", source="sox - -|")
    cut = SimpleNamespace(recording=SimpleNamespace(sources=[source]))
    bl._relocate_recording(cut, tmp_path)
    assert source.source == "sox - -|"


def _tiny_cutset(tmp_path, cut_id="MEET-0.0-2.0"):
    """Write a one-cut CutSet over a real two-second wav."""
    import numpy as np
    import soundfile as sf
    from lhotse import CutSet, MonoCut, Recording, SupervisionSegment

    audio_dir = tmp_path / "audio" / "MEET" / "audio"
    audio_dir.mkdir(parents=True)
    wav = audio_dir / "MEET.Array1-01.wav"
    sf.write(str(wav), np.zeros(32000, dtype="float32"), 16000)

    recording = Recording.from_file(str(wav), recording_id="MEET")
    cut = MonoCut(
        id=cut_id,
        start=0.0,
        duration=2.0,
        channel=0,
        recording=recording,
        supervisions=[
            SupervisionSegment(
                id="s1",
                recording_id="MEET",
                start=0.0,
                duration=0.5,
                text="hello",
                speaker="B",
            ),
            SupervisionSegment(
                id="s2",
                recording_id="MEET",
                start=1.0,
                duration=0.5,
                text="a much longer turn",
                speaker="A",
            ),
        ],
    )
    path = tmp_path / "cutsets" / "tiny.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    CutSet.from_cuts([cut]).to_file(str(path))
    return path


def _single_split_config(tmp_path, monkeypatch, **sot):
    """Point the module config at a one-split corpus under tmp_path."""
    options = {
        "cutset": "tiny.jsonl.gz",
        "ordering": "start_time",
        "lowercase": True,
        "max_timestamp_pause": 2.0,
    }
    options.update(sot)
    # The three location keys interpolate ${oc.env:...} lazily, so they are set
    # through the environment. Assigning over such a node freezes it: monkeypatch
    # restores the resolved string rather than the interpolation, and every later
    # test in the process would then read whatever root this one used.
    monkeypatch.setenv("AMI_SOT_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("AMI_SOT_CUTSET_DIR", "cutsets")
    monkeypatch.setenv("AMI_AUDIO_ROOT", "audio")
    monkeypatch.setitem(bl._CONFIG, "split_dirs", {"test": "data/test"})
    monkeypatch.setitem(bl._CONFIG, "sot", {"test": options})
    monkeypatch.setitem(bl._CONFIG, "separator", "????")
    monkeypatch.setitem(bl._CONFIG, "prompt", "<|en|><|transcribe|>")
    monkeypatch.setitem(bl._CONFIG, "na_symbol", "<|nospeech|>")


def _built_cuts(tmp_path, split="test"):
    """Return the cuts the builder wrote for one split."""
    from lhotse import CutSet

    path = tmp_path / "data" / split / bl._CUTS_FILENAME
    assert path.is_file(), path
    return list(CutSet.from_file(str(path)))


@pytest.mark.execution_timeout(30)
def test_build_writes_one_cutset_carrying_the_group_ids(tmp_path, monkeypatch):
    """A built split is one file, and every cut in it carries its own text.

    Ids and text cannot drift apart the way separate files could, so what is left
    to pin is that both are there.
    """
    _tiny_cutset(tmp_path)
    _single_split_config(tmp_path, monkeypatch)

    bl.AmiSotBuilder().build()

    cuts = _built_cuts(tmp_path)
    assert [cut.id for cut in cuts] == ["MEET-0.0-2.0"]
    assert all(cut.custom[bl._SOT_TEXT_FIELD] for cut in cuts)


@pytest.mark.execution_timeout(30)
def test_build_serializes_with_the_splits_configured_ordering(tmp_path, monkeypatch):
    """The ordering option must reach the text, not just the config."""
    _tiny_cutset(tmp_path)
    _single_split_config(tmp_path, monkeypatch, ordering="longest_first")

    bl.AmiSotBuilder().build()

    text = _built_cuts(tmp_path)[0].custom[bl._SOT_TEXT_FIELD]
    # "A" speaks later but has the longer block, so longest_first leads on it.
    assert text.startswith(
        "<|en|><|transcribe|><|1.00|> a much longer turn<|1.50|> ????"
    )
    assert not text.endswith("<|endoftext|>")


@pytest.mark.execution_timeout(30)
def test_build_leaves_the_group_audio_in_the_meeting_recording(tmp_path, monkeypatch):
    """No per-group wav is written, and the cut still yields the group's audio.

    Writing one file per group cost 8.6 GB across the three splits. The cut names
    the offset and duration instead, so the samples have to come out the same.
    """
    _tiny_cutset(tmp_path)
    _single_split_config(tmp_path, monkeypatch)

    bl.AmiSotBuilder().build()

    assert not (tmp_path / "data" / "test" / "segments_wav").exists()
    cut = _built_cuts(tmp_path)[0]
    assert cut.load_audio().shape[1] == 32000  # the cut's own 2.0 s at 16 kHz


@pytest.mark.execution_timeout(30)
def test_is_built_becomes_true_only_after_building(tmp_path, monkeypatch):
    _tiny_cutset(tmp_path)
    _single_split_config(tmp_path, monkeypatch)
    builder = bl.AmiSotBuilder()

    assert builder.is_built() is False
    assert builder.is_source_prepared() is True  # cutset and audio are present
    builder.build()
    assert builder.is_built() is True


@pytest.mark.execution_timeout(30)
def test_asking_whether_the_source_is_prepared_downloads_nothing(tmp_path, monkeypatch):
    """A question about disk must be answered from disk.

    BaseSystem.create_dataset asks this of roots that hold nothing. Answering by
    fetching turns the question into a multi-hundred-gigabyte AMI download, which
    is slow enough to look like a hang and fills the caller's disk before it fails.
    """
    import lhotse.recipes.ami as ami

    def _refuse(*args, **kwargs):
        raise AssertionError("is_source_prepared must not download")

    monkeypatch.setattr(ami, "download_audio", _refuse)
    _tiny_cutset(tmp_path)
    _single_split_config(tmp_path, monkeypatch)
    # Cutsets present, audio absent: the branch the download sits behind.
    monkeypatch.setenv("AMI_AUDIO_ROOT", "no_audio_here")

    assert bl.AmiSotBuilder().is_source_prepared() is False


@pytest.mark.execution_timeout(30)
def test_rebuilding_a_split_produces_the_same_cuts(tmp_path, monkeypatch):
    """Re-running the stage over a built split must not change what it holds."""
    _tiny_cutset(tmp_path)
    _single_split_config(tmp_path, monkeypatch)
    bl.AmiSotBuilder().build()
    first = [(cut.id, cut.custom[bl._SOT_TEXT_FIELD]) for cut in _built_cuts(tmp_path)]

    bl.AmiSotBuilder().build()

    assert [
        (cut.id, cut.custom[bl._SOT_TEXT_FIELD]) for cut in _built_cuts(tmp_path)
    ] == first


# --------------------------------------------------------------------------
# Regression guards for the serialization options.
#
# These values were recovered by reproducing egs2/ami/sot_asr1/data/*/text
# from the cutsets: train 52972/52972, dev 3746/3746, test 6127/6127, byte
# for byte. Editing one silently changes every reference the recipe scores
# against, so the committed values are asserted directly.
# --------------------------------------------------------------------------

_VERIFIED_OPTIONS = {
    "train": ("start_time", True, 2.0),
    "valid": ("start_time", True, 2.0),
    "test": ("longest_first", True, 2.0),
}


@pytest.mark.parametrize("split", sorted(_VERIFIED_OPTIONS))
def test_committed_sot_options_are_the_verified_ones(split):
    options = bl._CONFIG["sot"][split]
    assert (
        options["ordering"],
        options["lowercase"],
        options["max_timestamp_pause"],
    ) == _VERIFIED_OPTIONS[split]


def test_no_split_carries_its_own_separator():
    """No split carries its own separator.

    A token list cannot follow a per-split separator, and a model trained on one symbol
    cannot be scored against another.
    """
    for options in bl._CONFIG["sot"].values():
        assert "separator" not in options


# Needs both halves: the cutsets to rebuild from and the recorded transcript
# to rebuild against.
@pytest.mark.execution_timeout(600)
@ami_sot_paths.needs_cutsets
@ami_sot_paths.needs_reference_text
def test_test_split_transcripts_still_match_the_reference_content():
    """The text is S2T-shaped now, so compare what did not change.

    Stripping the prompt and rewriting the separator spelling must give back the
    prepared reference minus its end token, byte for byte. That keeps the guarantee on
    the part that matters -- the transcript and its timestamps -- while allowing the
    shape to change.
    """
    from lhotse import CutSet

    options = bl._CONFIG["sot"]["test"]
    try:
        normalizer = bl.get_text_norm(options.get("text_norm"))
    except ImportError as exc:
        # The configured normalizer is an optional dependency. Without it the
        # builder cannot produce the prepared text at all, so there is nothing
        # this test could compare.
        pytest.skip(str(exc).splitlines()[0])
    cuts = {
        c.id: c
        for c in CutSet.from_file(str(ami_sot_paths.CUTSET_DIR / options["cutset"]))
    }
    prompt = bl._CONFIG["prompt"]

    mismatched = []
    for line in ami_sot_paths.REFERENCE_TEXT.read_text().splitlines():
        utt_id, _, expected = line.partition(" ")
        cut = cuts.get(utt_id)
        assert cut is not None, f"{utt_id} is absent from the cutset"
        built = bl.sot_text.build_sot_text(
            cut.supervisions,
            max_timestamp_pause=options["max_timestamp_pause"],
            ordering=options["ordering"],
            separator=bl._CONFIG["separator"],
            lowercase=options["lowercase"],
            text_norm=normalizer,
            prompt=prompt,
            eos=None,
        )
        assert built.startswith(prompt), utt_id

        def normalise(text):
            # Compare transcripts, not spellings: either side may carry the
            # prompt, and the separator has two spellings in circulation.
            text = text.removeprefix(prompt).removesuffix(" <|endoftext|>")
            return text.replace(" ???? ", " <sc> ")

        if normalise(built) != normalise(expected):
            mismatched.append(utt_id)

    assert not mismatched, f"{len(mismatched)} differ, first: {mismatched[:3]}"


def test_data_config_reaches_the_builder_through_the_create_dataset_stage():
    """Data config reaches the builder through the create dataset stage.

    conf/training.yaml is the only CLI route to the builder, so it must resolve, and it
    must name all three splits.

    The create_dataset stage is gated on --training_config, so one config drives every
    stage, as it does in the other egs3 recipes. Training reads train and valid; the
    builder prepares test as well, and the inference stage assumes it was prepared.
    """
    from espnet3.components.data.dataset_module import (
        load_dataset_module,
        parse_dataset_reference_config,
    )
    from espnet3.utils.config_utils import load_config_with_defaults

    config = load_config_with_defaults(
        str(ami_sot_paths.RECIPE / "conf" / "training.yaml"), resolve=True
    )
    assert {"train", "valid", "test"} <= set(config["dataset"])
    # The same block carries the preprocessor the train stage instantiates.
    assert "preprocessor" in config["dataset"]

    # One shared data_src, so create_dataset prepares it once and build()
    # loops over the splits itself.
    sources = {
        parse_dataset_reference_config(dict(config["dataset"][split][0]))[0]
        for split in ("train", "valid", "test")
    }
    assert sources == {None}

    module = load_dataset_module(data_src=None, recipe_dir=str(ami_sot_paths.RECIPE))
    assert getattr(module, "DatasetBuilder") is not None
    assert module.DatasetBuilder.__name__ == "AmiSotBuilder"


@ami_sot_paths.needs_cutsets
@ami_sot_paths.needs_reference_text
@pytest.mark.execution_timeout(600)
def test_test_split_selects_exactly_the_reference_utterances():
    """The builder must emit the reference's ROW SET, not just matching text.

    Its sibling regression looks each reference line up in the cutset by id and compares
    the text, which says nothing about WHICH cuts the builder would write. The manifest
    holds 6137 cuts and the recorded reference holds 6127: ten utterance groups overran
    Whisper's 30 s window, were truncated to exactly 30.00 s, and are excluded. Without
    the max_cut_duration filter the builder emits all 6137 and every downstream count is
    wrong.
    """
    from lhotse import CutSet

    options = bl._CONFIG["sot"]["test"]
    max_duration = bl._CONFIG["max_cut_duration"]
    cuts = list(CutSet.from_file(str(ami_sot_paths.CUTSET_DIR / options["cutset"])))

    emitted = sorted(c.id for c in cuts if c.duration < max_duration)
    expected = sorted(
        line.split()[0]
        for line in ami_sot_paths.REFERENCE_TEXT.read_text().splitlines()
    )

    assert len(cuts) > len(expected), "the filter must actually drop something"
    assert emitted == expected


@pytest.mark.execution_timeout(30)
def test_build_refuses_a_corpus_it_did_not_create(tmp_path, monkeypatch):
    """The shared AMI corpus is writable and build() rewrites what it finds.

    Anything that flips is_built() to False -- a partially copied tree, a missing file
    -- would otherwise have create_dataset overwrite a corpus the recipe did not
    produce.
    """
    _tiny_cutset(tmp_path)
    _single_split_config(tmp_path, monkeypatch)
    split_dir = tmp_path / "data" / "test"
    split_dir.mkdir(parents=True)
    foreign = split_dir / bl._CUTS_FILENAME
    foreign.write_bytes(b"someone else's corpus")

    with pytest.raises(RuntimeError, match="not created by this builder"):
        bl.AmiSotBuilder().build()

    assert foreign.read_bytes() == b"someone else's corpus"


@pytest.mark.execution_timeout(30)
def test_prepare_source_refuses_a_corpus_it_did_not_create(tmp_path, monkeypatch):
    """create_dataset calls prepare_source itself, and before is_built.

    Guarding build() alone would still let an AMI download start inside a corpus root
    that is not ours.
    """
    _tiny_cutset(tmp_path)
    _single_split_config(tmp_path, monkeypatch)
    split_dir = tmp_path / "data" / "test"
    split_dir.mkdir(parents=True)
    (split_dir / bl._CUTS_FILENAME).write_bytes(b"someone else's corpus")

    with pytest.raises(RuntimeError, match="not created by this builder"):
        bl.AmiSotBuilder().prepare_source()

    assert not (tmp_path / "downloads").exists()


@pytest.mark.execution_timeout(30)
def test_build_proceeds_on_an_empty_root_and_leaves_its_marker(tmp_path, monkeypatch):
    _tiny_cutset(tmp_path)
    _single_split_config(tmp_path, monkeypatch)

    bl.AmiSotBuilder().build()

    assert (tmp_path / bl._MARKER_NAME).is_file()


@pytest.mark.execution_timeout(30)
def test_an_interrupted_build_can_be_resumed(tmp_path, monkeypatch):
    """An interrupted build can be resumed.

    The marker claims the directory up front, so a crash partway does not lock the
    builder out of its own half-written corpus.
    """
    _tiny_cutset(tmp_path)
    _single_split_config(tmp_path, monkeypatch)
    marker = tmp_path / bl._MARKER_NAME
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("claimed\n")
    split_dir = tmp_path / "data" / "test"
    split_dir.mkdir(parents=True)
    (split_dir / bl._CUTS_FILENAME).write_bytes(b"half-written")

    bl.AmiSotBuilder().build()

    assert [cut.id for cut in _built_cuts(tmp_path)] == ["MEET-0.0-2.0"]


@pytest.mark.execution_timeout(300)
def test_build_writes_a_token_list_the_separator_resolves_in(tmp_path, monkeypatch):
    """Build writes a token list the separator resolves in.

    S2TTask sizes the model from len(token_list); the separator must be in it exactly
    once, at the id the tokenizer emits.
    """
    _tiny_cutset(tmp_path)
    _single_split_config(tmp_path, monkeypatch)

    bl.AmiSotBuilder().build()

    tokens = (tmp_path / "data" / "tokens.txt").read_text().splitlines()
    assert tokens.count(bl._CONFIG["separator"]) == 1
    for symbol in ("<|0.00|>", "<|30.00|>", "<|nospeech|>"):
        assert symbol in tokens, symbol


@pytest.mark.execution_timeout(300)
def test_a_corpus_keeps_reporting_itself_built_without_a_token_list(
    tmp_path, monkeypatch
):
    """The opposite of the obvious design, and deliberate.

    Requiring the vocabulary here would make every corpus prepared before this recipe
    report itself unbuilt, which sends create_dataset into build() and straight into the
    guard. The owner of such a corpus needs one file added, not a refusal.
    """
    _tiny_cutset(tmp_path)
    _single_split_config(tmp_path, monkeypatch)
    bl.AmiSotBuilder().build()
    assert bl.AmiSotBuilder().is_built() is True

    (tmp_path / "data" / "tokens.txt").unlink()
    assert bl.AmiSotBuilder().is_built() is True


@pytest.mark.execution_timeout(300)
def test_write_token_list_adds_the_vocabulary_without_rebuilding(tmp_path, monkeypatch):
    """The supported route for a corpus that predates this recipe."""
    _tiny_cutset(tmp_path)
    _single_split_config(tmp_path, monkeypatch)
    bl.AmiSotBuilder().build()
    cuts_before = [
        (cut.id, cut.custom[bl._SOT_TEXT_FIELD]) for cut in _built_cuts(tmp_path)
    ]
    (tmp_path / "data" / "tokens.txt").unlink()

    bl.AmiSotBuilder().write_token_list()

    assert (tmp_path / "data" / "tokens.txt").is_file()
    assert [
        (cut.id, cut.custom[bl._SOT_TEXT_FIELD]) for cut in _built_cuts(tmp_path)
    ] == cuts_before
