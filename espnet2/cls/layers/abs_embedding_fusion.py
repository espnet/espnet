from abc import ABC, abstractmethod

import torch


class AbsEmbeddingFusion(torch.nn.Module, ABC):
    """
    Abstract base class defining the interface for combining two embeddings.

    Users should implement the `forward` method, which takes two embeddings and
    returns a combined embedding.
    """

    def forward(self, embeddings: dict, lengths: dict = None):
        """
        Combine two embeddings (of possibly varying shapes, types, etc.) into one
        embedding. Must be implemented by subclasses.

        Args:
            embeddings: A dict of embeddings to combine.
            lengths: A dict of lengths corresponding to the embeddings.
            Both dicts have the same keys, which are the names of the embeddings.

        Returns:
            A combined embedding, according to the logic implemented by the subclass.
        """
        raise NotImplementedError
