"""Build small deterministic TSE mixtures from the checked-in AN4 fixture."""

import json
from importlib import resources
from pathlib import Path

import numpy as np
import soundfile as sf

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import extract_targz


def fixture_config():
    """Load the fixture's corpus paths and small dataset limits."""
    with resources.as_file(
        resources.files(__package__).joinpath("config.yaml")
    ) as path:
        return load_config_with_defaults(str(path))


class MiniTSEBuilder(DatasetBuilder):
    """Prepare a bounded AN4 smoke fixture, with no network access."""

    def is_source_prepared(self, recipe_dir, **kwargs):
        """Check for the extracted speech source without modifying it."""
        return (Path(recipe_dir) / fixture_config().source).is_dir()

    def prepare_source(self, recipe_dir, **kwargs):
        """Extract the local archive supplied by CI; never download a corpus."""
        root = Path(recipe_dir)
        archive = root / fixture_config().archive
        if not archive.is_file():
            raise FileNotFoundError(
                f"Missing {archive}; copy mini_an4/esp2_asr/downloads.tar.gz here"
            )
        extract_targz(archive, root)

    def is_built(self, recipe_dir, **kwargs):
        """Check the task-ready split manifests."""
        cfg = fixture_config()
        return all(
            (Path(recipe_dir) / cfg.data / f"{split}.json").is_file()
            for split in cfg.splits
        )

    def build(self, recipe_dir, **kwargs):
        """Write tiny paired mixtures, enrollment waveforms and manifests."""
        cfg = fixture_config()
        root = Path(recipe_dir).resolve()
        paths = sorted((root / cfg.source).rglob("*.sph"))
        if len(paths) < 2:
            raise FileNotFoundError(
                "At least two AN4 source recordings are required; run prepare_source"
            )
        data = root / cfg.data
        data.mkdir(parents=True, exist_ok=True)
        for split, count in cfg.splits.items():
            rows = []
            for idx in range(count):
                waves = []
                for offset in (0, 1):
                    audio, rate = sf.read(
                        paths[(idx + offset) % len(paths)], dtype="float32"
                    )
                    if rate != cfg.sample_rate:
                        raise ValueError(
                            f"Fixture expects {cfg.sample_rate} Hz, got {rate}"
                        )
                    audio = audio[: cfg.max_samples]
                    waves.append(np.pad(audio, (0, cfg.max_samples - len(audio))))
                uid = f"{split}-{idx}"
                fields = {
                    "speech_mix": (waves[0] + waves[1]) / 2,
                    "speech_ref1": waves[0] / 2,
                    "speech_ref2": waves[1] / 2,
                    "enroll_ref1": waves[0],
                    "enroll_ref2": waves[1],
                }
                row = {"utt_id": uid}
                for key, audio in fields.items():
                    path = data / f"{uid}-{key}.wav"
                    sf.write(path, audio, cfg.sample_rate, subtype="FLOAT")
                    row[key] = str(path)
                rows.append(row)
            (data / f"{split}.json").write_text(json.dumps(rows), encoding="utf-8")
