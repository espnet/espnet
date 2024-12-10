from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.lm.abs_model import AbsLM
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class ESPnetLanguageModel(AbsESPnetModel):
    """
        The ESPnetLanguageModel class implements a language model using the ESPnet
    framework. It is designed to compute the negative log likelihood of sequences
    of text, enabling the training and evaluation of language models.

    Attributes:
        lm (AbsLM): An instance of a language model that follows the AbsLM
            interface.
        sos (int): The start-of-sequence token index.
        eos (int): The end-of-sequence token index.
        ignore_id (int): The token index to ignore during loss computation,
            default is 0, which may be shared with CTC-blank symbol for ASR.

    Args:
        lm (AbsLM): The language model instance to be used.
        vocab_size (int): The size of the vocabulary.
        ignore_id (int, optional): The index of the token to ignore. Default is 0.

    Methods:
        nll(text: torch.Tensor, text_lengths: torch.Tensor,
            max_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
            Computes the negative log likelihood of the input text.

        batchify_nll(text: torch.Tensor, text_lengths: torch.Tensor,
            batch_size: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
            Computes the negative log likelihood in batches to avoid out-of-memory
            errors.

        forward(text: torch.Tensor, text_lengths: torch.Tensor,
            **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
            Computes the forward pass and returns the loss, statistics, and weight.

        collect_feats(text: torch.Tensor, text_lengths: torch.Tensor,
            **kwargs) -> Dict[str, torch.Tensor]:
            Collects features from the input text. Currently, it returns an empty
            dictionary.

    Examples:
        # Create a language model
        lm = ESPnetLanguageModel(lm_model_instance, vocab_size=1000)

        # Compute negative log likelihood
        nll, lengths = lm.nll(text_tensor, text_lengths_tensor)

        # Compute batch negative log likelihood
        nll_batch, lengths_batch = lm.batchify_nll(text_tensor, text_lengths_tensor)

        # Forward pass
        loss, stats, weight = lm.forward(text_tensor, text_lengths_tensor)

    Note:
        This class assumes that the input tensors are on the same device as the
        model. It also assumes that the language model provided is compatible with
        the expected input format.
    """

    @typechecked
    def __init__(self, lm: AbsLM, vocab_size: int, ignore_id: int = 0):
        super().__init__()
        self.lm = lm
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1

        # ignore_id may be assumed as 0, shared with CTC-blank symbol for ASR.
        self.ignore_id = ignore_id

    def nll(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute negative log likelihood (nll).

        This function is typically called within the `batchify_nll` method to
        calculate the negative log likelihood for a given batch of text data.

        Args:
            text (torch.Tensor): A tensor of shape (Batch, Length) representing
                the input text sequences.
            text_lengths (torch.Tensor): A tensor of shape (Batch,) that contains
                the lengths of each text sequence in the batch.
            max_length (Optional[int]): An optional integer to limit the maximum
                length of the sequences. If None, it will use the maximum length
                from `text_lengths`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - A tensor of shape (Batch, Length) representing the negative
                  log likelihood for each sequence in the batch.
                - A tensor of shape (Batch,) representing the lengths of the
                  processed sequences after padding.

        Examples:
            >>> model = ESPnetLanguageModel(lm, vocab_size=100)
            >>> text = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> text_lengths = torch.tensor([3, 3])
            >>> nll, lengths = model.nll(text, text_lengths)
            >>> print(nll.shape)  # Output: (2, 4)
            >>> print(lengths)     # Output: tensor([4, 4])

        Note:
            This method uses padding to ensure that all sequences are of equal
            length for batch processing. The `<sos>` and `<eos>` tokens are
            used to denote the start and end of the sequences, respectively.

        Raises:
            ValueError: If the shapes of `text` and `text_lengths` do not
            align or if `max_length` is less than the maximum length in
            `text_lengths`.
        """
        batch_size = text.size(0)
        # For data parallel
        if max_length is None:
            text = text[:, : text_lengths.max()]
        else:
            text = text[:, :max_length]

        # 1. Create a sentence pair like '<sos> w1 w2 w3' and 'w1 w2 w3 <eos>'
        # text: (Batch, Length) -> x, y: (Batch, Length + 1)
        x = F.pad(text, [1, 0], "constant", self.eos)
        t = F.pad(text, [0, 1], "constant", self.ignore_id)
        for i, l in enumerate(text_lengths):
            t[i, l] = self.sos
        x_lengths = text_lengths + 1

        # 2. Forward Language model
        # x: (Batch, Length) -> y: (Batch, Length, NVocab)
        y, _ = self.lm(x, None)

        # 3. Calc negative log likelihood
        # nll: (BxL,)
        nll = F.cross_entropy(y.view(-1, y.shape[-1]), t.view(-1), reduction="none")
        # nll: (BxL,) -> (BxL,)
        if max_length is None:
            nll.masked_fill_(make_pad_mask(x_lengths).to(nll.device).view(-1), 0.0)
        else:
            nll.masked_fill_(
                make_pad_mask(x_lengths, maxlen=max_length + 1).to(nll.device).view(-1),
                0.0,
            )
        # nll: (BxL,) -> (B, L)
        nll = nll.view(batch_size, -1)
        return nll, x_lengths

    def batchify_nll(
        self, text: torch.Tensor, text_lengths: torch.Tensor, batch_size: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Compute negative log likelihood (nll) from transformer language model.

        To avoid Out Of Memory (OOM) errors, this function separates the input into
        batches. It then calls the `nll` method for each batch and combines the results
        before returning them.

        Attributes:
            None

        Args:
            text: A tensor of shape (Batch, Length) containing the input text data.
            text_lengths: A tensor of shape (Batch,) indicating the lengths of each
                sequence in the batch.
            batch_size: An integer specifying the number of samples each batch
                contains when computing nll. Adjust this value to avoid OOM errors
                or to increase efficiency.

        Returns:
            A tuple containing:
                - nll: A tensor of shape (Total, Length) with the computed negative
                    log likelihood for each sequence.
                - x_lengths: A tensor of shape (Total,) with the lengths of each
                    processed sequence.

        Examples:
            >>> model = ESPnetLanguageModel(lm, vocab_size=100)
            >>> text = torch.randint(0, 100, (250, 50))
            >>> text_lengths = torch.randint(1, 51, (250,))
            >>> nll, x_lengths = model.batchify_nll(text, text_lengths, batch_size=100)

        Note:
            The method uses the `nll` function to compute negative log likelihood for
            each batch of text, which is crucial for language model training.
        """
        total_num = text.size(0)
        if total_num <= batch_size:
            nll, x_lengths = self.nll(text, text_lengths)
        else:
            nlls = []
            x_lengths = []
            max_length = text_lengths.max()

            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_text = text[start_idx:end_idx, :]
                batch_text_lengths = text_lengths[start_idx:end_idx]
                # batch_nll: [B * T]
                batch_nll, batch_x_lengths = self.nll(
                    batch_text, batch_text_lengths, max_length=max_length
                )
                nlls.append(batch_nll)
                x_lengths.append(batch_x_lengths)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nlls)
            x_lengths = torch.cat(x_lengths)
        assert nll.size(0) == total_num
        assert x_lengths.size(0) == total_num
        return nll, x_lengths

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
            Forward pass for the ESPnetLanguageModel.

        This method computes the negative log likelihood (NLL) of the input text
        and returns the loss, statistics, and the weight for the current batch.
        It first calls the `nll` method to obtain the NLL and the lengths of the
        output tokens, then calculates the loss as the sum of the NLL divided
        by the total number of tokens.

        Args:
            text (torch.Tensor): Input tensor of shape (Batch, Length) representing
                                 the sequences of text.
            text_lengths (torch.Tensor): Tensor of shape (Batch,) containing the
                                          lengths of each sequence in the batch.
            **kwargs: Additional keyword arguments (unused in this method).

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]: A tuple
            containing:
                - loss (torch.Tensor): The computed loss for the batch.
                - stats (Dict[str, torch.Tensor]): A dictionary containing statistics
                  such as the loss.
                - weight (torch.Tensor): The number of tokens processed in the batch.

        Examples:
            >>> model = ESPnetLanguageModel(lm, vocab_size=1000)
            >>> text = torch.randint(0, 999, (32, 10))  # Batch of 32 sequences of length 10
            >>> text_lengths = torch.randint(1, 11, (32,))  # Random lengths for each sequence
            >>> loss, stats, weight = model.forward(text, text_lengths)
            >>> print(loss, stats, weight)

        Note:
            The loss is computed in a way that handles padding and ensures that
            the model can be used in a data-parallel setting.
        """
        nll, y_lengths = self.nll(text, text_lengths)
        ntokens = y_lengths.sum()
        loss = nll.sum() / ntokens
        stats = dict(loss=loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, ntokens), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
            Collect features from the input text tensor.

        This method processes the input text and text lengths to collect features,
        which can be useful for various downstream tasks. The specific implementation
        details depend on the language model being used. Currently, this method
        returns an empty dictionary.

        Args:
            text (torch.Tensor): A tensor of shape (Batch, Length) containing the input
                text data.
            text_lengths (torch.Tensor): A tensor of shape (Batch,) containing the
                lengths of each input text in the batch.
            **kwargs: Additional keyword arguments that may be used by subclasses.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the collected features.

        Examples:
            >>> model = ESPnetLanguageModel(lm, vocab_size=100)
            >>> text = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> text_lengths = torch.tensor([3, 3])
            >>> features = model.collect_feats(text, text_lengths)
            >>> print(features)  # Output: {}
        """
        return {}
