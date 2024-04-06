from typing import Dict, Tuple, Union

from espnet2.train.dataset import AbsDataset


class ESPnetEZDataset(AbsDataset):
    def __init__(self, dataset, data_info, phase=None):
        self.dataset = dataset
        self.data_info = data_info
        self.phase = phase

    def has_name(self, name) -> bool:
        if self.phase is not None:
            return name in self.data_info[self.phase]
        else:
            return name in self.data_info

    def names(self) -> Tuple[str, ...]:
        if self.phase is not None:
            return tuple(self.data_info[self.phase].keys())
        else:
            return tuple(self.data_info.keys())

    def __getitem__(self, uid: Union[str, int]) -> Tuple[str, Dict]:
        idx = int(uid)
        if self.phase is not None:
            data_info = self.data_info[self.phase]
        else:
            data_info = self.data_info

        return (
            str(uid),
            {k: v(self.dataset[idx]) for k, v in data_info.items()},
        )

    def __len__(self) -> int:
        return len(self.dataset)
