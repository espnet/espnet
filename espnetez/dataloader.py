
from espnet2.iterators.abs_iter_factory import AbsIterFactory
from torch.utils.data import DataLoader


class Dataloader(AbsIterFactory):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def build_iter(self, epoch: int, shuffle: bool = None) -> DataLoader:
        return DataLoader(**self.kwargs)
