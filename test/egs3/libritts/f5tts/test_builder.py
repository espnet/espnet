"""Tests for the LibriTTS builder's LibriSpeech-PC eval-manifest wiring.

The default eval config (`conf/inference.yaml`) reads
`data/librispeech_pc/manifest.tsv`, so `create_dataset` has to produce it.
These tests pin that contract, and pin that `build()` never touches the
network: downloads belong to `prepare_source()` alone.
"""

import socket
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import pytest

from egs3.libritts.f5tts.dataset import builder as builder_module
from egs3.libritts.f5tts.dataset.builder import (
    _ARCHIVE_SIZES,
    _CFG,
    LibriTTSBuilder,
    _download_subset,
)
from espnet3.components.data.dataset_module import _load_local_dataset_module

RECIPE = Path(__file__).resolve().parents[4] / "egs3" / "libritts" / "f5tts"

LST_ROW = (
    "4992-41806-0009\t4.355\texclaimed Bill Harmon to his wife.\t"
    "4992-23283-0000\t6.645\tBut the more forgetfulness had then prevailed.\n"
)

LIBRITTS_SUBSETS = [
    subset for subsets in _CFG["split_subsets"].values() for subset in subsets
]
LSPC_CFG = _CFG["librispeech_pc"]


def _make_libritts(recipe_dir: Path) -> None:
    """Create the LibriTTS subset directories, empty but marked complete.

    `build()` raises on a missing subset directory, and empty subsets simply
    yield empty split manifests, which is all these tests need. The
    `.complete` marker is what `is_source_prepared` looks at.
    """
    for subset in LIBRITTS_SUBSETS:
        subset_dir = recipe_dir / _CFG["dataset_path"] / "LibriTTS" / subset
        subset_dir.mkdir(parents=True, exist_ok=True)
        (subset_dir / ".complete").touch()


def _make_librispeech(recipe_dir: Path) -> Path:
    """Create a LibriSpeech test-clean tree holding the two fixture flacs."""
    root = recipe_dir / _CFG["dataset_path"] / LSPC_CFG["test_clean_path"]
    for utt in ("4992-41806-0009", "4992-23283-0000"):
        spk, chap, _ = utt.split("-")
        d = root / spk / chap
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{utt}.flac").write_bytes(b"fake")
    (root / ".complete").touch()
    return root


def _make_lst(recipe_dir: Path) -> Path:
    """Write a one-row stand-in for the F5-TTS cross-sentence pair list."""
    lst = recipe_dir / _CFG["dataset_path"] / LSPC_CFG["lst_path"]
    lst.parent.mkdir(parents=True, exist_ok=True)
    lst.write_text(LST_ROW, encoding="utf-8")
    return lst


def _make_libritts_manifests(recipe_dir: Path) -> None:
    """Write the three LibriTTS split manifests, empty but present."""
    for relpath in _CFG["manifest_paths"].values():
        path = recipe_dir / _CFG["data_path"] / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


@pytest.fixture()
def recipe_dir(tmp_path: Path) -> Path:
    """Return a recipe root with both corpora and the pair list in place."""
    _make_libritts(tmp_path)
    _make_librispeech(tmp_path)
    _make_lst(tmp_path)
    return tmp_path


@pytest.fixture()
def no_network(monkeypatch):
    """Make any network access from the code under test raise loudly.

    `socket.socket` is the floor every in-process stdlib networking path
    reaches, and `subprocess.run` covers the out-of-process shape the download
    scripts use - monkeypatching sockets alone would not see a wget spawned in
    a child process.
    """

    def _blocked(*_args, **_kwargs):
        raise AssertionError("network access is not allowed here")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(urllib.request, "urlopen", _blocked)
    monkeypatch.setattr(subprocess, "run", _blocked)


def test_is_built_false_without_librispeech_pc_manifest(tmp_path):
    """The LibriTTS manifests alone must not count as built.

    Without this, create_dataset reports success while the default eval config
    has nothing to read.
    """
    _make_libritts_manifests(tmp_path)
    assert not LibriTTSBuilder().is_built(recipe_dir=tmp_path)

    manifest = tmp_path / _CFG["data_path"] / LSPC_CFG["manifest_path"]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("", encoding="utf-8")
    assert LibriTTSBuilder().is_built(recipe_dir=tmp_path)


def test_training_is_not_blocked_by_a_missing_eval_manifest(tmp_path):
    """Training must not depend on the LibriSpeech-PC eval manifest.

    LibriTTSDataset guards on is_libritts_built, not is_built. If it guarded on
    is_built, any checkout whose LibriTTS manifests were built before the eval
    manifest joined create_dataset would refuse to start training until the
    user downloaded 346 MB of LibriSpeech that training never reads.
    """
    _make_libritts_manifests(tmp_path)
    builder = LibriTTSBuilder()

    # Eval manifest deliberately absent.
    assert not builder.is_built(recipe_dir=tmp_path)
    assert builder.is_libritts_built(recipe_dir=tmp_path)


def test_is_libritts_built_false_when_a_split_manifest_is_missing(tmp_path):
    _make_libritts_manifests(tmp_path)
    (tmp_path / _CFG["data_path"] / _CFG["manifest_paths"]["valid"]).unlink()
    assert not LibriTTSBuilder().is_libritts_built(recipe_dir=tmp_path)


def test_build_writes_librispeech_pc_manifest(recipe_dir, no_network):
    """build() writes the eval manifest, and does so without any network I/O."""
    LibriTTSBuilder().build(recipe_dir=recipe_dir)

    manifest = recipe_dir / _CFG["data_path"] / LSPC_CFG["manifest_path"]
    assert manifest.is_file()
    rows = manifest.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    gen_utt, gen_text, ref_utt, ref_wav, ref_text = rows[0].split("\t")
    assert gen_utt == "4992-23283-0000"
    assert gen_text == "But the more forgetfulness had then prevailed."
    assert ref_utt == "4992-41806-0009"
    assert ref_wav == str(
        recipe_dir
        / _CFG["dataset_path"]
        / LSPC_CFG["test_clean_path"]
        / "4992"
        / "41806"
        / "4992-41806-0009.flac"
    )
    assert ref_text == "exclaimed Bill Harmon to his wife."

    # The LibriTTS manifests are still written by the same call.
    for relpath in _CFG["manifest_paths"].values():
        assert (recipe_dir / _CFG["data_path"] / relpath).is_file()

    assert LibriTTSBuilder().is_built(recipe_dir=recipe_dir)


def test_build_raises_when_lst_missing(recipe_dir, no_network):
    """A missing pair list fails loudly instead of downloading from build()."""
    (recipe_dir / _CFG["dataset_path"] / LSPC_CFG["lst_path"]).unlink()
    with pytest.raises(FileNotFoundError, match="pair list"):
        LibriTTSBuilder().build(recipe_dir=recipe_dir)


def test_is_source_prepared_requires_librispeech_and_lst(tmp_path):
    """A LibriTTS tree on its own is not a prepared source for this recipe."""
    builder = LibriTTSBuilder()

    _make_libritts(tmp_path)
    assert not builder.is_source_prepared(recipe_dir=tmp_path)

    # LibriSpeech present, pair list still missing.
    _make_librispeech(tmp_path)
    assert not builder.is_source_prepared(recipe_dir=tmp_path)

    # Pair list present, but LibriSpeech lacks its `.complete` marker: the
    # directory alone could be the partial tree an interrupted extraction
    # leaves behind.
    lst = _make_lst(tmp_path)
    test_clean = tmp_path / _CFG["dataset_path"] / LSPC_CFG["test_clean_path"]
    (test_clean / ".complete").unlink()
    assert not builder.is_source_prepared(recipe_dir=tmp_path)

    (test_clean / ".complete").touch()
    assert lst.is_file()
    assert builder.is_source_prepared(recipe_dir=tmp_path)


def test_prepare_source_is_a_noop_when_everything_is_present(recipe_dir, no_network):
    """A fully prepared tree re-runs without downloading anything."""
    LibriTTSBuilder().prepare_source(recipe_dir=recipe_dir)


def test_build_through_the_stage_loader(recipe_dir, no_network):
    """The manifest is written when the builder is loaded the way run.py loads it.

    The training config carries no `data_src`, so `create_dataset` loads
    `dataset/__init__.py` from its file path under a synthetic module name
    rather than as `egs3.libritts.f5tts.dataset`; the relative import of
    `librispeech_pc` inside the builder has to resolve under that name too.
    """
    module = _load_local_dataset_module(RECIPE)
    assert module.__name__.startswith("_espnet3_local_dataset_")

    module.DatasetBuilder().build(recipe_dir=recipe_dir)

    manifest = recipe_dir / _CFG["data_path"] / LSPC_CFG["manifest_path"]
    assert manifest.read_text(encoding="utf-8").count("\n") == 1


def _make_archive(path: Path, corpus: str, subset: str, size: int) -> None:
    """Write a `<corpus>/<subset>/` tarball padded to the published size."""
    inner = path.parent / "_archive_src" / corpus / subset
    inner.mkdir(parents=True)
    (inner / "README").write_text("x", encoding="utf-8")
    with tarfile.open(path, "w:gz") as tar:
        tar.add(inner.parent.parent, arcname=".")
    with path.open("ab") as f:
        f.write(b"\0" * (size - path.stat().st_size))


def test_download_subset_skips_a_completed_subset(tmp_path, no_network):
    """The `.complete` marker alone decides; nothing is fetched or extracted."""
    marker = tmp_path / "LibriTTS" / "dev-clean" / ".complete"
    marker.parent.mkdir(parents=True)
    marker.touch()

    _download_subset(tmp_path, "dev-clean")


def test_download_subset_rejects_unknown_subsets(tmp_path, no_network):
    with pytest.raises(ValueError, match="Unknown LibriTTS subset"):
        _download_subset(tmp_path, "train-clean-999")
    with pytest.raises(ValueError, match="Unknown LibriSpeech subset"):
        _download_subset(tmp_path, "dev-clean", corpus="LibriSpeech")


def test_download_subset_reuses_a_complete_archive(tmp_path, no_network):
    """An archive of the published size is extracted, not downloaded again."""
    archive = tmp_path / "test-clean.tar.gz"
    _make_archive(
        archive, "LibriTTS", "test-clean", _ARCHIVE_SIZES["LibriTTS"]["test-clean"]
    )

    _download_subset(tmp_path, "test-clean")

    assert (tmp_path / "LibriTTS" / "test-clean" / ".complete").is_file()
    assert (tmp_path / "LibriTTS" / "test-clean" / "README").is_file()
    assert archive.is_file()


def test_download_subset_refetches_a_partial_archive(tmp_path, monkeypatch):
    """A wrong-sized archive is removed and the subset is downloaded afresh."""
    archive = tmp_path / "LibriSpeech_test-clean.tar.gz"
    archive.write_bytes(b"partial")
    fetched = []

    def fake_download_url(url, dst_path, logger=None):
        fetched.append(url)
        _make_archive(
            dst_path,
            "LibriSpeech",
            "test-clean",
            _ARCHIVE_SIZES["LibriSpeech"]["test-clean"],
        )

    monkeypatch.setattr(builder_module, "download_url", fake_download_url)

    _download_subset(tmp_path, "test-clean", corpus="LibriSpeech", remove_archive=True)

    # LibriSpeech archives carry a prefix: both corpora publish test-clean.tar.gz.
    assert fetched == ["https://www.openslr.org/resources/12/test-clean.tar.gz"]
    assert (tmp_path / "LibriSpeech" / "test-clean" / ".complete").is_file()
    assert not archive.exists()


def test_prepare_source_downloads_what_is_missing(tmp_path, monkeypatch):
    """Every corpus and the pair list go through the same two helpers."""
    calls = []
    monkeypatch.setattr(
        builder_module,
        "_download_subset",
        lambda root, subset, corpus="LibriTTS", remove_archive=False: calls.append(
            (corpus, subset, remove_archive)
        ),
    )
    monkeypatch.setattr(
        builder_module,
        "_download_pair_list",
        lambda url, dest: calls.append(("lst", url, str(dest))),
    )

    LibriTTSBuilder().prepare_source(recipe_dir=tmp_path, remove_archive=True)

    assert calls[: len(LIBRITTS_SUBSETS)] == [
        ("LibriTTS", subset, True) for subset in LIBRITTS_SUBSETS
    ]
    assert calls[len(LIBRITTS_SUBSETS)] == ("LibriSpeech", LSPC_CFG["subset"], True)
    assert calls[-1][:2] == ("lst", LSPC_CFG["lst_url"])


def test_lst_url_is_pinned_to_a_commit():
    """A moving ref would silently change this third-party eval set."""
    url = LSPC_CFG["lst_url"]
    assert url.endswith("/librispeech_pc_test_clean_cross_sentence.lst")
    sha = url.split("/F5-TTS/")[1].split("/")[0]
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)
