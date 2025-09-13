"""Provides a factory for creating PyTorch DataLoader instances.

The :class:`~espnet2.iterators.dataloader.Dataloader` class is a concrete
implementation of :class:`~espnet2.iterators.abs_iter_factory.AbsIterFactory`
that stores keyword arguments to be passed to
:class:`torch.utils.data.DataLoader`.  It exposes a :py:meth:`build_iter`
method which returns a ready-to-use DataLoader for a given epoch.

Typical usage::

    from espnet2.iterators.dataloader import Dataloader
    from torch.utils.data import Dataset

    class MyDataset(Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return idx

    loader = Dataloader(dataset=MyDataset(), batch_size=16, shuffle=True)
    train_loader = loader.build_iter(epoch=1)
    for batch in train_loader:
        # process batch

The ``epoch`` argument is currently unused but is included for future
epoch-dependent behaviour such as shuffling.  The ``shuffle`` parameter
is also ignored in the current implementation, keeping the interface
compatible with potential future extensions.

Dependencies
------------
- `torch` - provides :class:`torch.utils.data.DataLoader`.
- :class:`espnet2.iterators.abs_iter_factory.AbsIterFactory` - abstract
  factory base class.
"""

from torch.utils.data import DataLoader

from espnet2.iterators.abs_iter_factory import AbsIterFactory


class Dataloader(AbsIterFactory):
    """DataLoader class for building data iterators.

    This class extends the `AbsIterFactory` and provides a method to create
    a PyTorch DataLoader. It accepts keyword arguments that can be passed
    directly to the DataLoader constructor.

    Attributes:
        kwargs (dict): A dictionary of keyword arguments for the DataLoader.

    Args:
        **kwargs: Arbitrary keyword arguments that are passed to the
            `torch.utils.data.DataLoader` constructor. Common arguments include:
            - dataset: The dataset from which to load the data.
            - batch_size: Number of samples per batch.
            - shuffle: Whether to shuffle the data at every epoch.
            - num_workers: Number of subprocesses to use for data loading.
            - collate_fn: A function to merge a list of samples to form a
                mini-batch.

    Examples:
        >>> from torchvision import datasets, transforms
        >>> transform = transforms.Compose([transforms.ToTensor()])
        >>> dataset = datasets.MNIST(root='./data', train=True,
        ...                            transform=transform, download=True)
        >>> dataloader = Dataloader(dataset=dataset, batch_size=32,
        ...                          shuffle=True)
        >>> train_loader = dataloader.build_iter(epoch=1)

    Note:
        The `build_iter` method does not utilize the `epoch` and `shuffle`
        parameters in its current implementation. They are included for
        potential future use, such as enabling shuffling based on epoch
        number.
    """

    def __init__(self, **kwargs):
        """Initialize the instance, storing arbitrary keyword arguments.

        Args:
            kwargs: Arbitrary keyword arguments to be stored on the instance.

        The keyword arguments are stored in the :attr:`kwargs` attribute for
        later use.
        """
        self.kwargs = kwargs

    def build_iter(self, epoch: int, shuffle: bool = None) -> DataLoader:
        """Construct a PyTorch DataLoader instance.

        This method initializes and returns a DataLoader using the
        parameters provided during the instantiation of the Dataloader
        class. It can be customized with various options to suit the
        needs of the data loading process.

        Args:
            epoch (int): The current epoch number. This can be used
                to implement epoch-based behaviors, such as shuffling
                the dataset.
            shuffle (bool, optional): A flag indicating whether to
                shuffle the dataset. If set to True, the data will be
                shuffled before each epoch. Defaults to None, which
                means the value set in kwargs will be used.

        Returns:
            DataLoader: A PyTorch DataLoader instance configured with
            the provided arguments.

        Examples:
            >>> dataloader = Dataloader(batch_size=32, dataset=my_dataset)
            >>> train_loader = dataloader.build_iter(epoch=1, shuffle=True)
            >>> for data in train_loader:
            ...     # Process data
            ...     pass

        Note:
            Ensure that the dataset provided in kwargs is compatible
            with the DataLoader's expected format.
        """
        return DataLoader(**self.kwargs)
