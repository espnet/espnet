"""Read a manifest produced by the recipe's raw-corpus preparation scripts."""

import logging
from pathlib import Path

from egs3.voxlingua107.lid.dataset.dataset import VoxLingua107Dataset, _read_manifest


class Dataset(VoxLingua107Dataset):
    """Read prepared speech, optionally restricting evaluation to known languages."""

    def __init__(self, manifest, lang2utt=None):
        """Use the training inventory for closed-set evaluation when provided."""
        self.examples = _read_manifest(Path(manifest))
        self.sample_rate = 16000
        self.speed_perturb_factors = (1.0,)
        if lang2utt is not None:
            with Path(lang2utt).open(encoding="utf-8") as source:
                languages = {line.split()[0] for line in source if line.strip()}
            total = len(self.examples)
            self.examples = [e for e in self.examples if e.language in languages]
            logging.info(
                "%s: retaining %d/%d utterances in the training language inventory",
                manifest,
                len(self.examples),
                total,
            )
