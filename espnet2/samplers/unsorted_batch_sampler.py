import logging
from typing import Iterator, Optional, Tuple

from typeguard import typechecked

from espnet2.fileio.read_text import read_2columns_text
from espnet2.samplers.abs_sampler import AbsSampler


class UnsortedBatchSampler(AbsSampler):
    """
    UnsortedBatchSampler is a BatchSampler that generates batches of a constant
    size without performing any sorting. It is particularly useful in decoding
    mode or for tasks that do not involve sequence-to-sequence learning, such
    as classification.

    This class does not require length information as it directly uses the keys
    from the provided key file to create batches.

    Attributes:
        batch_size (int): The size of each batch.
        key_file (str): The path to the key file containing the utterances.
        drop_last (bool): Whether to drop the last incomplete batch.
        batch_list (list): A list of batches created from the key file.

    Args:
        batch_size (int): The size of each batch. Must be greater than 0.
        key_file (str): The path to the key file.
        drop_last (bool, optional): If True, drop the last incomplete batch.
            Defaults to False.
        utt2category_file (str, optional): An optional file mapping utterances
            to categories. If provided, must match the keys in the key file.

    Raises:
        RuntimeError: If the key file is empty or if there is a mismatch between
            the keys in the key file and the categories in the
            utt2category_file.

    Examples:
        >>> sampler = UnsortedBatchSampler(batch_size=2, key_file='keys.txt')
        >>> for batch in sampler:
        ...     print(batch)

    Note:
        The keys in the key file should be formatted such that each line contains
        an utterance key. If a category file is provided, it should also have the
        same keys for proper mapping.

    Todo:
        - Add support for dynamic batch sizes based on category distribution.
    """

    @typechecked
    def __init__(
        self,
        batch_size: int,
        key_file: str,
        drop_last: bool = False,
        utt2category_file: Optional[str] = None,
    ):
        assert batch_size > 0
        self.batch_size = batch_size
        self.key_file = key_file
        self.drop_last = drop_last

        # utt2shape:
        #    uttA <anything is o.k>
        #    uttB <anything is o.k>
        utt2any = read_2columns_text(key_file)
        if len(utt2any) == 0:
            logging.warning(f"{key_file} is empty")
        # In this case the, the first column in only used
        keys = list(utt2any)
        if len(keys) == 0:
            raise RuntimeError(f"0 lines found: {key_file}")

        category2utt = {}
        if utt2category_file is not None:
            utt2category = read_2columns_text(utt2category_file)
            if set(utt2category) != set(keys):
                raise RuntimeError(
                    f"keys are mismatched between {utt2category_file} != {key_file}"
                )
            for k, v in utt2category.items():
                category2utt.setdefault(v, []).append(k)
        else:
            category2utt["default_category"] = keys

        self.batch_list = []
        for d, v in category2utt.items():
            category_keys = v
            # Apply max(, 1) to avoid 0-batches
            N = max(len(category_keys) // batch_size, 1)
            if not self.drop_last:
                # Split keys evenly as possible as. Note that If N != 1,
                # the these batches always have size of batch_size at minimum.
                cur_batch_list = [
                    category_keys[i * len(keys) // N : (i + 1) * len(keys) // N]
                    for i in range(N)
                ]
            else:
                cur_batch_list = [
                    tuple(category_keys[i * batch_size : (i + 1) * batch_size])
                    for i in range(N)
                ]
            self.batch_list.extend(cur_batch_list)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
            f"key_file={self.key_file}, "
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)
