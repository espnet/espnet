from torch.utils.data import DataLoader

from espnet2.iterators.abs_iter_factory import AbsIterFactory


class Dataloader(AbsIterFactory):
    """
    DataLoader class for building data iterators.

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
        self.kwargs = kwargs

    def build_iter(self, epoch: int, shuffle: bool = None) -> DataLoader:
        """
        Constructs a PyTorch DataLoader instance.

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
