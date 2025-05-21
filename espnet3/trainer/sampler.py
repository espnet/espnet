from typing import Sequence

from espnet2.samplers.abs_sampler import AbsSampler


class MappedSamplerWrapper(AbsSampler):
    def __init__(self, base_sampler: AbsSampler, shape_files: Sequence[str]):
        self.base_sampler = base_sampler

        self.utt2idx = {}
        for s in shape_files:
            with open(s, encoding="utf-8") as f:
                for line in f:
                    uttid = line.strip().split()[0]
                    if uttid in self.utt2idx:
                        raise ValueError(f"Duplicate uttid found: {uttid}")
                    self.utt2idx[uttid] = len(self.utt2idx)

    def __len__(self):
        return len(self.base_sampler)

    def __iter__(self):
        for batch in self.base_sampler:
            yield tuple(self.utt2idx[utt] for utt in batch)

    def generate(self, seed):
        return [
            tuple(self.utt2idx[utt] for utt in batch)
            for batch in self.base_sampler.generate(seed)
        ]
