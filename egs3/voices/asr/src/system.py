"""Statistics adapter preserving ESPnet2 raw-input numel batching."""

from copy import deepcopy
from pathlib import Path

from hydra.utils import instantiate

from egs3.voices.asr.dataset.builder import write_text
from espnet3.systems.asr.system import ASRSystem


class VoicesSystem(ASRSystem):
    """Use standard ASR stages with raw-speech and token shape files."""

    def collect_stats(self, *args, **kwargs):
        """Preserve GlobalMVN and write the exact ESPnet2 numel sampler inputs.

        The upstream statistics stage mutates normalization configuration and
        emits feature shapes only. The source recipe batches raw waveform
        lengths and token lengths, appending vocabulary size to text shapes.
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
        vocabulary = len(Path(original.model.token_list).read_text().splitlines())
        for split in ("train", "valid"):
            shapes = {"speech": [], "text": []}
            dataset = getattr(organizer, split)
            for index in range(len(dataset)):
                sample = dataset[index]
                shapes["speech"].append(f"{index} {len(sample['speech'])}\n")
                shapes["text"].append(f"{index} {len(sample['text'])},{vocabulary}\n")
            for key, rows in shapes.items():
                write_text(
                    Path(original.stats_dir) / split / f"{key}_shape", "".join(rows)
                )
