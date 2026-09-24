"""Read ID-prefixed LM text for the existing ESPnet3 DataOrganizer."""

from array import array
from pathlib import Path

from torch.utils.data import Dataset as TorchDataset


class LMTextDataset(TorchDataset):
    """Index text by byte offset without holding the full LM corpus in memory.

    Args:
        text_path: UTF-8 file with one ``utterance_id transcript`` per line.

    Examples:
        >>> dataset = LMTextDataset("data/lm/train.txt")
        >>> sample = dataset[0]  # Only the text field reaches the preprocessor.
    """

    def __init__(self, text_path):
        """Index nonempty transcripts in their original file order.

        Args:
            text_path: Existing ID-prefixed text file; do not modify it in use.

        Returns:
            None. Only byte offsets are retained in memory.

        Raises:
            FileNotFoundError: The text file does not exist.
            ValueError: No nonempty transcripts are available.

        Examples:
            >>> dataset = LMTextDataset("data/lm/train.txt")
        """
        self.text_path = Path(text_path).resolve()
        self.offsets = array("Q")
        with self.text_path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if len(line.split(maxsplit=1)) == 2:
                    self.offsets.append(offset)
        if not self.offsets:
            raise ValueError(f"No nonempty transcripts in {self.text_path}")

    def __len__(self):
        """Return the number of indexed transcripts.

        Args:
            None.

        Returns:
            Number of nonempty utterances.

        Examples:
            >>> count = len(dataset)
        """
        return len(self.offsets)

    def __getitem__(self, index):
        """Read a transcript without passing its original ID into the model.

        Args:
            index: Integer position in the indexed file.

        Returns:
            Dictionary containing a UTF-8 ``text`` string.

        Raises:
            IndexError: The index is outside the dataset.

        Examples:
            >>> dataset[0]
            {'text': 'HELLO WORLD'}
        """
        # Independent handles also work in spawned/forked dataloader workers.
        with self.text_path.open("rb") as stream:
            stream.seek(self.offsets[index])
            line = stream.readline().decode("utf-8").rstrip("\r\n")
        return {"text": line.split(maxsplit=1)[1]}


# Dataset references resolve this conventional export through DataOrganizer.
Dataset = LMTextDataset
