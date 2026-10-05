"""Tests for the LibriTTS recipe's builder (resampling) and dataset adapter."""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from omegaconf import OmegaConf

from egs3.libritts.esp2_gan_tts.dataset import builder as builder_module
from egs3.libritts.esp2_gan_tts.dataset.builder import LibriTTSBuilder
from egs3.libritts.esp2_gan_tts.dataset.dataset import LibriTTSDataset
from espnet3.utils.config_utils import load_default_config

SOURCE_FS = 24000
TARGET_FS = builder_module._CFG["fs"]


def _make_libritts_tree(recipe_root: Path, seconds: float = 0.1) -> dict[str, Path]:
    """Create one 24 kHz utterance per configured subset, plus the markers."""
    libritts_root = recipe_root / builder_module._CFG["dataset_path"] / "LibriTTS"
    wavs = {}
    for idx, subset in enumerate(builder_module._required_subsets()):
        speaker, chapter = str(100 + idx), str(200 + idx)
        utt_id = f"{speaker}_{chapter}_000001_000000"
        utt_dir = libritts_root / subset / speaker / chapter
        utt_dir.mkdir(parents=True)
        wav_path = utt_dir / f"{utt_id}.wav"
        t = np.arange(int(seconds * SOURCE_FS)) / SOURCE_FS
        sf.write(wav_path, 0.5 * np.sin(2 * np.pi * 440 * t), SOURCE_FS, "PCM_16")
        (utt_dir / f"{utt_id}.normalized.txt").write_text(
            f"Hello from {subset}.", encoding="utf-8"
        )
        (libritts_root / subset / ".complete").touch()
        wavs[subset] = wav_path
    return wavs


def test_is_source_prepared_requires_markers(tmp_path: Path) -> None:
    builder = LibriTTSBuilder()
    assert builder.is_source_prepared(recipe_dir=tmp_path) is False

    _make_libritts_tree(tmp_path)
    assert builder.is_source_prepared(recipe_dir=tmp_path) is True

    # A bare directory without its marker is an interrupted extraction.
    first = builder_module._required_subsets()[0]
    (tmp_path / "downloads" / "LibriTTS" / first / ".complete").unlink()
    assert builder.is_source_prepared(recipe_dir=tmp_path) is False


def test_build_resamples_audio_and_writes_manifests(tmp_path: Path) -> None:
    """build() writes a 22.05 kHz PCM_16 copy and manifests that point at it."""
    sources = _make_libritts_tree(tmp_path)
    builder = LibriTTSBuilder()
    assert builder.is_built(recipe_dir=tmp_path) is False

    builder.build(recipe_dir=tmp_path)

    assert builder.is_built(recipe_dir=tmp_path) is True
    data_dir = tmp_path / builder_module._CFG["data_path"]
    audio_root = data_dir / builder_module._CFG["audio_path"]
    assert (audio_root / ".complete").is_file()

    for split, relpath in builder_module._CFG["manifest_paths"].items():
        rows = (data_dir / relpath).read_text(encoding="utf-8").splitlines()
        assert len(rows) == len(builder_module._CFG["split_subsets"][split])
        for row in rows:
            utt_id, wav_path, text, sid = row.split("\t")
            wav_path = Path(wav_path)
            # The resampled copy mirrors the corpus layout under data/wav.
            assert wav_path.is_relative_to(audio_root)
            assert wav_path.name == f"{utt_id}.wav"
            info = sf.info(str(wav_path))
            assert info.samplerate == TARGET_FS
            assert info.subtype == "PCM_16"
            assert info.channels == 1
            source = next(p for p in sources.values() if p.name == wav_path.name)
            expected = round(sf.info(str(source)).frames * TARGET_FS / SOURCE_FS)
            assert abs(info.frames - expected) <= 1
            assert text.startswith("Hello from ")
            assert sid.isdigit()

    # The originals are untouched and still 24 kHz.
    assert all(sf.info(str(p)).samplerate == SOURCE_FS for p in sources.values())


def test_build_resumes_without_rewriting(tmp_path: Path) -> None:
    _make_libritts_tree(tmp_path)
    builder = LibriTTSBuilder()
    builder.build(recipe_dir=tmp_path)
    audio_root = tmp_path / "data" / builder_module._CFG["audio_path"]
    written = sorted(p for p in audio_root.rglob("*.wav"))
    mtimes = {p: p.stat().st_mtime_ns for p in written}
    (audio_root / ".complete").unlink()

    builder.build(recipe_dir=tmp_path)

    assert {p: p.stat().st_mtime_ns for p in written} == mtimes
    assert (audio_root / ".complete").is_file()
    assert not list(audio_root.rglob("*.tmp"))


def test_build_honours_fs_override(tmp_path: Path) -> None:
    _make_libritts_tree(tmp_path)
    LibriTTSBuilder().build(recipe_dir=tmp_path, fs=16000)

    audio_root = tmp_path / "data" / builder_module._CFG["audio_path"]
    rates = {sf.info(str(p)).samplerate for p in audio_root.rglob("*.wav")}
    assert rates == {16000}


def test_dataset_reads_resampled_audio(tmp_path: Path) -> None:
    _make_libritts_tree(tmp_path)
    LibriTTSBuilder().build(recipe_dir=tmp_path)

    dataset = LibriTTSDataset(
        "valid", recipe_dir=tmp_path, load_xvector=False, inference=True
    )
    sample = dataset[0]

    assert set(sample) == {"text", "speech", "utt_id", "wav_path", "raw_text"}
    assert sample["speech"].dtype == np.float32
    assert sample["speech"].ndim == 1
    assert abs(len(sample["speech"]) - round(0.1 * TARGET_FS)) <= 1
    assert str(sample["utt_id"]) == "103_203_000001_000000"


def test_dataset_resamples_mismatched_audio_with_warning(
    tmp_path: Path, caplog
) -> None:
    """A manifest still pointing at the 24 kHz originals is resampled on the fly.

    Same semantics as the F5-TTS LibriTTS dataset: ``fs`` is a target rate,
    not an assertion. The warning fires once so the slow path is visible.
    """
    _make_libritts_tree(tmp_path)
    LibriTTSBuilder().build(recipe_dir=tmp_path)
    originals = sorted((tmp_path / "downloads").rglob("*.wav"))[:2]
    manifest = tmp_path / "stale.tsv"
    manifest.write_text(
        "".join(f"u{i}\t{p}\thello\t0\n" for i, p in enumerate(originals)),
        encoding="utf-8",
    )
    dataset = LibriTTSDataset(
        "train", recipe_dir=tmp_path, manifest_path=manifest, load_xvector=False
    )

    with caplog.at_level(logging.WARNING):
        first = dataset[0]["speech"]
        second = dataset[1]["speech"]

    assert abs(len(first) - round(0.1 * TARGET_FS)) <= 1
    assert abs(len(second) - round(0.1 * TARGET_FS)) <= 1
    assert first.dtype == np.float32
    warnings = [r for r in caplog.records if "resampling to 22050 Hz" in r.message]
    assert len(warnings) == 1

    # fs=None keeps each file's own rate, like the F5 dataset's default.
    native = LibriTTSDataset(
        "train",
        recipe_dir=tmp_path,
        manifest_path=manifest,
        load_xvector=False,
        fs=None,
    )
    assert native[0]["speech"].shape == (round(0.1 * SOURCE_FS),)


def test_template_preprocessor_rate_matches_builder() -> None:
    """One rate for the data (builder) and the preprocessor (template)."""
    cfg = load_default_config("training.yaml", "egs3.TEMPLATE.esp2_gan_tts")
    assert cfg.dataset.preprocessor.fs == TARGET_FS
    inference = load_default_config("inference.yaml", "egs3.TEMPLATE.esp2_gan_tts")
    assert inference.dataset.preprocessor.fs == TARGET_FS


def test_recipe_model_rates_match_builder() -> None:
    """The recipe's mel loss, sampling_rate and output wav rate all agree."""
    recipe_conf = Path(__file__).resolve().parents[4] / (
        "egs3/libritts/esp2_gan_tts/conf"
    )
    training = OmegaConf.load(recipe_conf / "training.yaml")
    inference = OmegaConf.load(recipe_conf / "inference.yaml")
    assert training.model.tts_conf.mel_loss_params.fs == TARGET_FS
    assert training.model.tts_conf.sampling_rate == TARGET_FS
    assert inference.output_artifacts.wav.sample_rate == TARGET_FS
