import torch

class TransformDataset(torch.utils.data.Dataset):
    ''' Transform Dataset
    '''
    def __init__(self, data, transform):
        super(TransformDataset).__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx])


class ChainerDataLoader(object):
    def __init__(self, **kwargs):
        self.loader = torch.utils.data.dataloader.DataLoader(**kwargs)
        self.len = len(kwargs['dataset'])
        self.current_position = 0
        self.epoch = 0
        self.iter = None
        self.kwargs = kwargs

    def next(self):
        if self.iter is None:
            self.iter = iter(self.loader)
        try:
            ret = next(self.iter)
        except StopIteration:
            self.iter = None
            return self.next()
        self.current_position += 1
        if self.current_position == self.len:
            self.epoch = self.epoch + 1
            self.current_position = 0
        return ret

    def __iter__(self):
        for batch in self.loader:
            yield batch

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / self.len

    def serialize(self, serializer):
        epoch = serializer('epoch', self.epoch)
        current_position = serializer('current_position', self.current_position)
        self.epoch = epoch
        self.current_position = current_position

    def start_shuffle(self):
        self.kwargs['shuffle'] = True
        self.loader = torch.utils.data.dataloader.DataLoader(**self.kwargs)

    def finalize(self):
        del self.loader
