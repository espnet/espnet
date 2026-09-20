"""Small statistics adapter for the source recipe's folded batching."""

from copy import deepcopy
from pathlib import Path

from hydra.utils import instantiate

from egs3.an4.asr.dataset.builder import atomic_write
from espnet3.systems.asr.system import ASRSystem


class An4System(ASRSystem):
    """Use the standard ASR stages with input lengths for folded sampling."""

    def collect_stats(self, *args, **kwargs):
        """Collect features and raw-speech/token lengths without changing the model.

        ESPnet3 currently emits feature shapes only, while ESPnet2's folded
        sampler uses speech samples and token counts. Its statistics function
        also removes normalization from the supplied model config in place.
        Work on a copy so a following train stage retains GlobalMVN.
        """
        self._reject_stage_args("collect_stats", args, kwargs)
        self.train_tokenizer()
        original = self.training_config
        try:
            self.training_config = deepcopy(original)
            super().collect_stats()
        finally:
            self.training_config = original
        organizer = instantiate(original.dataset)
        for split in ("train", "valid"):
            dataset = getattr(organizer, split)
            shapes = {"speech": [], "text": []}
            for index in range(len(dataset)):
                sample = dataset[index]
                for key in shapes:
                    shapes[key].append(f"{index} {len(sample[key])}\n")
            for key, lines in shapes.items():
                atomic_write(
                    Path(original.stats_dir) / split / f"{key}_shape", "".join(lines)
                )
