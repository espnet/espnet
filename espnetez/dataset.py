from typing import Dict, Tuple, Union

from espnet2.train.dataset import AbsDataset


class ESPnetEZDataset(AbsDataset):
    def __init__(self, dataset, data_info):
        self.dataset = dataset
        self.data_info = data_info

    def has_name(self, name) -> bool:
        return name in self.data_info

    def names(self) -> Tuple[str, ...]:
        return tuple(self.data_info.keys())

    def __getitem__(self, uid: Union[str, int]) -> Tuple[str, Dict]:
        idx = int(uid)
        return (
            str(uid),
            {k: v(self.dataset[idx]) for k, v in self.data_info.items()},
        )

    def __len__(self) -> int:
        return len(self.dataset)
