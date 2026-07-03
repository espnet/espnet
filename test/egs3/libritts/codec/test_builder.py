from pathlib import Path

import numpy as np
import soundfile as sf

from egs3.libritts.codec.dataset.builder import LibriTTSCodecBuilder

# ===============================================================
# Test Case Summary for LibriTTSCodecBuilder
# ===============================================================
#
# | Test Name                                    | Description                                                            | # noqa: E501
# |------------------------------------------------|--------------------------------------------------------------------------| # noqa: E501
# | test_build_produces_hyphenated_utt_ids        | Manifest utt_ids are hyphenated, not LibriTTS's native underscore form   | # noqa: E501
# | test_hyphenated_utt_ids_are_not_valid_int_literals | The hyphenated form can never be misparsed by Python's int()        | # noqa: E501


def _make_fake_libritts(tmp_path: Path) -> Path:
    """Build a tiny fake LibriTTS tree with all configured subsets present."""
    for subset in ["train-clean-100", "train-clean-360", "train-other-500"]:
        (tmp_path / "downloads" / "LibriTTS" / subset / "19" / "198").mkdir(
            parents=True, exist_ok=True
        )

    populated = tmp_path / "downloads" / "LibriTTS" / "train-clean-100" / "19" / "198"
    utt_id = "19_198_000000_000000"
    (populated / f"{utt_id}.normalized.txt").write_text("hello world")
    sf.write(
        str(populated / f"{utt_id}.wav"),
        np.zeros(1600, dtype=np.float32),
        16000,
    )

    for name in ["dev-clean", "test-clean"]:
        d = tmp_path / "downloads" / "LibriTTS" / name / "19" / "198"
        d.mkdir(parents=True, exist_ok=True)
        d_utt_id = f"19_198_0000_{name}"
        (d / f"{d_utt_id}.normalized.txt").write_text("hi")
        sf.write(str(d / f"{d_utt_id}.wav"), np.zeros(1600, dtype=np.float32), 16000)

    return tmp_path


def test_build_produces_hyphenated_utt_ids(tmp_path: Path) -> None:
    """Manifest utt_ids use hyphens (LibriSpeech-style), not raw LibriTTS underscores.

    ESPnet's chunk/sequence iterators index datasets by utt_id string keys via
    espnet3.components.data.dataset.CombinedDataset, which tries int(idx) on
    string keys before falling back to utterance-ID lookup. Python's int()
    accepts PEP 515 underscore-grouped digits, so LibriTTS's native
    underscore-separated utt_id (e.g. "19_198_000000_000000") would be
    silently misparsed as a huge integer instead of used as a lookup key.
    """
    recipe_root = _make_fake_libritts(tmp_path)
    builder = LibriTTSCodecBuilder()
    builder.build(recipe_dir=recipe_root)

    manifest = recipe_root / "data" / "manifest" / "train.tsv"
    lines = manifest.read_text().strip().splitlines()
    assert len(lines) == 1
    utt_id, _wav_path = lines[0].split("\t")
    assert "_" not in utt_id
    assert utt_id == "19-198-000000-000000"


def test_hyphenated_utt_ids_are_not_valid_int_literals(tmp_path: Path) -> None:
    """Hyphenated utt_ids always raise on int(), unlike the underscore form."""
    recipe_root = _make_fake_libritts(tmp_path)
    builder = LibriTTSCodecBuilder()
    builder.build(recipe_dir=recipe_root)

    manifest = recipe_root / "data" / "manifest" / "train.tsv"
    utt_id = manifest.read_text().strip().splitlines()[0].split("\t")[0]

    # Sanity: the underscore form is exactly what Python's int() accidentally
    # accepts (PEP 515 digit-group underscores), which is the bug this
    # hyphenation avoids.
    int(utt_id.replace("-", "_"))  # must not raise

    try:
        int(utt_id)
        raised = False
    except ValueError:
        raised = True
    assert raised, f"expected int({utt_id!r}) to raise ValueError"
