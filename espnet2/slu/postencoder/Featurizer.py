from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class Featurizer(nn.Module):
    """
    Featurizer take the :obj:`S3PRLUpstream`'s multiple layer of hidden_states and
    reduce (standardize) them into a single hidden_states, to connect with downstream NNs.

    This basic Featurizer expects all the layers to have same stride and hidden_size
    When the input upstream only have a single layer of hidden states, use that directly.
    If multiple layers are presented, add a trainable weighted-sum on top of those layers.

    Args:
        num_layers:
            int
        layer_selections (List[int]):
            To select a subset of hidden states from the given upstream by layer ids (0-index)
            If None (default), than all the layer of hidden states are selected
        normalize (bool):
            Whether to apply layer norm on all the hidden states before weighted-sum
            This can help convergence in some cases, but not used in SUPERB to ensure the
            fidelity of each upstream's extracted representation.

    Example::

        >>> import torch
        >>> from s3prl.nn import S3PRLUpstream, Featurizer
        ...
        >>> model = S3PRLUpstream("hubert")
        >>> model.eval()
        ...
        >>> with torch.no_grad():
        ...     wavs = torch.randn(2, 16000 * 2)
        ...     wavs_len = torch.LongTensor([16000 * 1, 16000 * 2])
        ...     all_hs, all_hs_len = model(wavs, wavs_len)
        ...
        >>> featurizer = Featurizer(model)
        >>> hs, hs_len = featurizer(all_hs, all_hs_len)
        ...
        >>> assert isinstance(hs, torch.FloatTensor)
        >>> assert isinstance(hs_len, torch.LongTensor)
        >>> batch_size, max_seq_len, hidden_size = hs.shape
        >>> assert hs_len.dim() == 1
    """

    def __init__(
        self,
        num_layers: int,
        output_size: int,
        layer_selections: List[int] = None,
        normalize: bool = False,
    ):
        super().__init__()
        self._output_size = output_size
        self.normalize = normalize

        if num_layers > 1:
            if layer_selections is not None:
                assert num_layers >= len(layer_selections)
                self.layer_selections = sorted(layer_selections)
            else:
                self.layer_selections = list(range(num_layers))
            self.weights = nn.Parameter(torch.zeros(len(self.layer_selections)))

    @property
    def output_size(self) -> int:
        """
        The hidden size of the final weighted-sum output
        """
        return self._output_size

    def _weighted_sum(self, all_hs, all_lens):
        assert len(all_hs) == len(all_lens) > 1
        for l in all_lens[1:]:
            torch.allclose(all_lens[0], l)
        stacked_hs = torch.stack(all_hs, dim=0)

        if self.normalize:
            stacked_hs = F.layer_norm(stacked_hs, (stacked_hs.shape[-1],))

        _, *origin_shape = stacked_hs.shape
        stacked_hs = stacked_hs.view(len(self.layer_selections), -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        weighted_hs = (norm_weights.unsqueeze(-1) * stacked_hs).sum(dim=0)
        weighted_hs = weighted_hs.view(*origin_shape)

        return weighted_hs, all_lens[0]

    def forward(
        self, all_hs: List[torch.FloatTensor], all_lens: List[torch.LongTensor]
    ):
        """
        Args:
            all_hs (List[torch.FloatTensor]): List[ (batch_size, seq_len, hidden_size) ]
            all_lens (List[torch.LongTensor]): List[ (batch_size, ) ]

        Return:
            torch.FloatTensor, torch.LongTensor

            1. The weighted-sum result, (batch_size, seq_len, hidden_size)
            2. the valid length of the result, (batch_size, )
        """
        if len(all_hs) == 1:
            return all_hs[0], all_lens[0]

        all_hs = [h for idx, h in enumerate(all_hs) if idx in self.layer_selections]
        all_lens = [l for idx, l in enumerate(all_lens) if idx in self.layer_selections]
        hs, hs_len = self._weighted_sum(all_hs, all_lens)
        return hs, hs_len
