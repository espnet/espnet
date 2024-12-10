import logging
from typing import Iterator, Tuple

from typeguard import typechecked

from espnet2.fileio.read_text import load_num_sequence_text
from espnet2.samplers.abs_sampler import AbsSampler


class SortedBatchSampler(AbsSampler):
    """
    BatchSampler with sorted samples by length.

    This sampler is designed to create batches of samples sorted by their
    lengths. It can sort samples either in ascending or descending order
    within a batch and allows for sorting of batches as well.

    Attributes:
        batch_size (int): The size of each batch.
        shape_file (str): Path to the file containing the shape information
            for each sample.
        sort_in_batch (str): Defines the sorting order for samples within
            each batch. Can be 'descending', 'ascending', or None.
        sort_batch (str): Defines the sorting order for the batches. Can be
            'ascending' or 'descending'.
        drop_last (bool): If True, drop the last batch if it is smaller than
            the batch size.

    Args:
        batch_size (int): The size of each batch. Must be greater than 0.
        shape_file (str): Path to the file containing the shape information
            for each sample.
        sort_in_batch (str): 'descending', 'ascending' or None for sorting
            samples within each batch.
        sort_batch (str): 'ascending' or 'descending' for sorting the batches.
        drop_last (bool): If True, the last batch will be dropped if it has
            fewer than batch_size samples.

    Raises:
        ValueError: If `sort_in_batch` or `sort_batch` is not one of the
            expected values.
        RuntimeError: If no samples or batches are found.

    Examples:
        >>> sampler = SortedBatchSampler(batch_size=32, shape_file='shapes.csv',
        ...                               sort_in_batch='ascending',
        ...                               sort_batch='descending')
        >>> for batch in sampler:
        ...     print(batch)

    Note:
        The shape file should be a CSV file where each line corresponds to a
        sample and contains its length.

    Todo:
        - Add functionality to handle different file formats for shape_file.
    """

    @typechecked
    def __init__(
        self,
        batch_size: int,
        shape_file: str,
        sort_in_batch: str = "descending",
        sort_batch: str = "ascending",
        drop_last: bool = False,
    ):
        assert batch_size > 0
        self.batch_size = batch_size
        self.shape_file = shape_file
        self.sort_in_batch = sort_in_batch
        self.sort_batch = sort_batch
        self.drop_last = drop_last

        # utt2shape: (Length, ...)
        #    uttA 100,...
        #    uttB 201,...
        utt2shape = load_num_sequence_text(shape_file, loader_type="csv_int")
        if sort_in_batch == "descending":
            # Sort samples in descending order (required by RNN)
            keys = sorted(utt2shape, key=lambda k: -utt2shape[k][0])
        elif sort_in_batch == "ascending":
            # Sort samples in ascending order
            keys = sorted(utt2shape, key=lambda k: utt2shape[k][0])
        else:
            raise ValueError(
                f"sort_in_batch must be either one of "
                f"ascending, descending, or None: {sort_in_batch}"
            )
        if len(keys) == 0:
            raise RuntimeError(f"0 lines found: {shape_file}")

        # Apply max(, 1) to avoid 0-batches
        N = max(len(keys) // batch_size, 1)
        if not self.drop_last:
            # Split keys evenly as possible as. Note that If N != 1,
            # the these batches always have size of batch_size at minimum.
            self.batch_list = [
                keys[i * len(keys) // N : (i + 1) * len(keys) // N] for i in range(N)
            ]
        else:
            self.batch_list = [
                tuple(keys[i * batch_size : (i + 1) * batch_size]) for i in range(N)
            ]

        if len(self.batch_list) == 0:
            logging.warning(f"{shape_file} is empty")

        if sort_in_batch != sort_batch:
            if sort_batch not in ("ascending", "descending"):
                raise ValueError(
                    f"sort_batch must be ascending or descending: {sort_batch}"
                )
            self.batch_list.reverse()

        if len(self.batch_list) == 0:
            raise RuntimeError("0 batches")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
            f"shape_file={self.shape_file}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch})"
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)
