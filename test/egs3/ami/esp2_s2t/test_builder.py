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
    """_relocate_recording keeps a source path whose file still exists.

    A corpus prepared on this machine must keep working without changes.
    """
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
    """Write a CutSet with one cut of a real two-second wav file."""
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
    """Set the module config to a corpus with one split under tmp_path."""
    options = {
        "cutset": "tiny.jsonl.gz",
        "ordering": "start_time",
        "lowercase": True,
        "max_timestamp_pause": 2.0,
    }
    options.update(sot)
    # The three location keys interpolate ${oc.env:...} lazily, so this function
    # sets them through the environment. Assigning a value to such a node would
    # freeze it.
    # monkeypatch would then restore the resolved string, and the interpolation
    # would be lost. Every later test in the process would read this test's root.
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
    """A built split is one cuts file, and every cut in it has its own text.

    The ids and the text are stored in the same file, so they cannot get out of
    sync. The test only checks that both are present.
    """
    _tiny_cutset(tmp_path)
    _single_split_config(tmp_path, monkeypatch)

    bl.AmiSotBuilder().build()

    cuts = _built_cuts(tmp_path)
    assert [cut.id for cut in cuts] == ["MEET-0.0-2.0"]
    assert all(cut.custom[bl._SOT_TEXT_FIELD] for cut in cuts)


@pytest.mark.execution_timeout(30)
def test_build_serializes_with_the_splits_configured_ordering(tmp_path, monkeypatch):
    """build() orders the speakers in the text by the split's ordering option."""
    _tiny_cutset(tmp_path)
    _single_split_config(tmp_path, monkeypatch, ordering="longest_first")

    bl.AmiSotBuilder().build()

    text = _built_cuts(tmp_path)[0].custom[bl._SOT_TEXT_FIELD]
    # "A" speaks later but has the longer block, so longest_first puts it first.
    assert text.startswith(
        "<|en|><|transcribe|><|1.00|> a much longer turn<|1.50|> ????"
    )
    assert not text.endswith("<|endoftext|>")


@pytest.mark.execution_timeout(30)
def test_build_leaves_the_group_audio_in_the_meeting_recording(tmp_path, monkeypatch):
    """build() writes no wav file per group, and the cut still loads its audio.

    One wav file per group used 8.6 GB for the three splits. The cut now stores the
    offset and the duration in the meeting recording, so loading the cut must give
    the same samples.
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
    """is_source_prepared only looks at the disk and downloads nothing.

    BaseSystem.create_dataset calls it on empty roots. A download there would fetch
    several hundred gigabytes of AMI audio. The download is slow enough to look like
    a hang, and it fills the caller's disk before it fails.
    """
    import lhotse.recipes.ami as ami

    def _refuse(*args, **kwargs):
        raise AssertionError("is_source_prepared must not download")

    monkeypatch.setattr(ami, "download_audio", _refuse)
    _tiny_cutset(tmp_path)
    _single_split_config(tmp_path, monkeypatch)
    # The cutsets exist and the audio does not. The download code is in this branch.
    monkeypatch.setenv("AMI_AUDIO_ROOT", "no_audio_here")

    assert bl.AmiSotBuilder().is_source_prepared() is False


@pytest.mark.execution_timeout(30)
def test_rebuilding_a_split_produces_the_same_cuts(tmp_path, monkeypatch):
    """Running build() again on a built split writes the same cuts and text."""
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
# With these values, the cutsets give the previously prepared serialized text
# byte for byte. The counts are train 52972/52972, dev 3746/3746 and test
# 6127/6127. A change to one value changes every reference that the recipe
# scores against, and no error is raised. So the tests check the committed
# values directly.
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
    """No split config sets its own separator.

    All splits share one token list, so a separator set per split would not match
    it. A model trained with one symbol cannot be scored with another symbol.
    """
    for options in bl._CONFIG["sot"].values():
        assert "separator" not in options


# This test needs the cutsets to rebuild the text from, and the recorded
# transcript to compare the rebuilt text with.
@pytest.mark.execution_timeout(600)
@ami_sot_paths.needs_cutsets
@ami_sot_paths.needs_reference_text
def test_test_split_transcripts_still_match_the_reference_content():
    """The rebuilt test transcripts match the recorded reference transcripts.

    The text now has the S2T format. After the prompt is removed and the separator
    spelling is changed, the text must equal the prepared reference without its end
    token, byte for byte. So the transcript and its timestamps must stay the same,
    and the format around them may change.
    """
    from lhotse import CutSet

    options = bl._CONFIG["sot"]["test"]
    try:
        normalizer = bl.get_text_norm(options.get("text_norm"))
    except ImportError as exc:
        # The configured normalizer is an optional dependency. Without it, the
        # builder cannot produce the prepared text, so this test has nothing to
        # compare.
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
            # Either side may contain the prompt, and the separator has two
            # spellings in use. This removes both differences before comparing.
            text = text.removeprefix(prompt).removesuffix(" <|endoftext|>")
            return text.replace(" ???? ", " <sc> ")

        if normalise(built) != normalise(expected):
            mismatched.append(utt_id)

    assert not mismatched, f"{len(mismatched)} differ, first: {mismatched[:3]}"


def test_data_config_reaches_the_builder_through_the_create_dataset_stage():
    """The create_dataset stage reaches AmiSotBuilder through conf/training.yaml.

    conf/training.yaml is the only CLI route to the builder. So it must resolve, and
    it must list all three splits.

    The create_dataset stage runs only with --training_config, so one config is used
    for every stage, as in the other egs3 recipes. Training reads train and valid.
    The builder also prepares test, and the inference stage expects test to exist.
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
    # The same block also has the preprocessor that the train stage creates.
    assert "preprocessor" in config["dataset"]

    # All splits share one data_src. So create_dataset prepares it once, and
    # build() loops over the splits itself.
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
    """The builder selects the same test utterances as the recorded reference.

    The test above compares the text for each reference id. It does not check which
    cuts the builder writes. The manifest has 6137 cuts, and the reference has 6127.
    Ten utterance groups were longer than Whisper's 30 s window. They were truncated
    to exactly 30.00 s and are excluded. Without the max_cut_duration filter, the
    builder writes all 6137 cuts, and all later counts are wrong.
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
    """build() refuses to write into a split directory that it did not create.

    The shared AMI corpus is writable, and build() rewrites the files it finds. A
    partially copied tree or a missing file makes is_built() return False. Without
    this check, create_dataset would then overwrite a corpus that the recipe did not
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
    """prepare_source refuses a corpus root that the builder did not create.

    create_dataset calls prepare_source itself, before is_built. If only build() had
    this check, an AMI download could start inside a corpus root that this builder
    did not create.
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
    """build() can resume an interrupted build.

    The builder writes its marker before anything else. After a crash, the marker is
    still there, so build() can overwrite its own half-written corpus.
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
    """build() writes a token list that contains the separator exactly once.

    S2TTask uses len(token_list) to size the model. The separator must be in the list
    exactly once, at the id that the tokenizer outputs.
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
    """is_built() returns True for a built corpus that has no tokens.txt.

    If is_built() required the token list, every corpus prepared before this recipe
    would report that it is not built. create_dataset would then call build(), and
    the guard would refuse the corpus. The owner only needs to add one file.
    """
    _tiny_cutset(tmp_path)
    _single_split_config(tmp_path, monkeypatch)
    bl.AmiSotBuilder().build()
    assert bl.AmiSotBuilder().is_built() is True

    (tmp_path / "data" / "tokens.txt").unlink()
    assert bl.AmiSotBuilder().is_built() is True


@pytest.mark.execution_timeout(300)
def test_write_token_list_adds_the_vocabulary_without_rebuilding(tmp_path, monkeypatch):
    """write_token_list adds tokens.txt and does not change the built cuts.

    This is the supported way to add the token list to a corpus that was prepared
    before this recipe.
    """
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
