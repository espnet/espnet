import torch

from espnet2.asvspoof.loss.abs_loss import AbsASVSpoofLoss


class ASVSpoofOCSoftmaxLoss(AbsASVSpoofLoss):
    """
    Implementation of the One-Class Softmax Loss for ASV Spoofing.

    This loss function is designed to differentiate between real and spoofed
    audio samples in the context of anti-spoofing systems. It utilizes a
    one-class softmax approach to handle embeddings from a speaker verification
    model.

    Attributes:
        weight (float): The weight of the loss. Default is 1.0.
        feat_dim (int): The dimension of the encoder's output features.
            Default is 128.
        m_real (float): Margin for real embeddings. Default is 0.5.
        m_fake (float): Margin for fake embeddings. Default is 0.2.
        alpha (float): Scaling factor for the loss. Default is 20.0.
        center (torch.nn.Parameter): Learnable parameter representing the
            center of the embedding space.
        softplus (torch.nn.Softplus): Softplus activation function.

    Args:
        weight (float, optional): Weight of the loss function. Defaults to 1.0.
        enc_dim (int, optional): Dimension of the encoder's output.
            Defaults to 128.
        m_real (float, optional): Margin for real embeddings. Defaults to 0.5.
        m_fake (float, optional): Margin for fake embeddings. Defaults to 0.2.
        alpha (float, optional): Scaling factor for the loss. Defaults to 20.0.

    Returns:
        None

    Raises:
        ValueError: If the input dimensions do not match the expected sizes.

    Examples:
        >>> loss_fn = ASVSpoofOCSoftmaxLoss()
        >>> labels = torch.tensor([[1], [0]])
        >>> embeddings = torch.randn(2, 10, 128)  # Batch of 2, 10 frames, 128 dim
        >>> loss = loss_fn(labels, embeddings)
        >>> print(loss)

    Note:
        The forward method computes the loss based on the provided labels and
        embeddings. The score method can be used to obtain the prediction
        scores for the embeddings.

    Todo:
        - Implement the loss computation in the forward method.
        - Complete the score method to return meaningful scores.
    """

    def __init__(
        self,
        weight: float = 1.0,
        enc_dim: int = 128,
        m_real: float = 0.5,
        m_fake: float = 0.2,
        alpha: float = 20.0,
    ):
        super(ASVSpoofOCSoftmaxLoss).__init__()
        self.weight = weight
        self.feat_dim = enc_dim
        self.m_real = m_real
        self.m_fake = m_fake
        self.alpha = alpha
        self.center = torch.nn.Parameter(torch.randn(1, self.feat_dim))
        torch.nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = torch.nn.Softplus()

    def forward(self, label: torch.Tensor, emb: torch.Tensor, **kwargs):
        """
        Compute the forward pass for the ASVSpoofOCSoftmaxLoss.

        This method calculates the loss based on the ground truth labels and
        the encoder embedding output. The embeddings are normalized, and scores
        are computed using the learned center and the embeddings.

        Args:
            label (torch.Tensor): Ground truth label tensor of shape
                [Batch, 1]. It indicates whether the input is real or spoofed.
            emb (torch.Tensor): Encoder embedding output tensor of shape
                [Batch, T, enc_dim]. This is the output from the encoder.

        Returns:
            torch.Tensor: The computed loss value for the input batch.

        Raises:
            ValueError: If the dimensions of `label` and `emb` do not match.

        Examples:
            >>> import torch
            >>> loss_fn = ASVSpoofOCSoftmaxLoss()
            >>> labels = torch.tensor([[1], [0]])  # Real and spoofed
            >>> embeddings = torch.randn(2, 10, 128)  # Example embeddings
            >>> loss = loss_fn.forward(labels, embeddings)
            >>> print(loss)

        Note:
            The loss computation involves several steps that include
            normalizing the embeddings and center, calculating scores,
            applying a bias, and using the Softplus function.

        Todo:
            - Implement the score computation based on the normalized embeddings.
            - Calculate the score bias based on `m_real` and `m_fake`.
            - Apply the `alpha` scaling and the Softplus activation.
            - Return the final computed loss value.
        """
        emb = torch.mean(emb, dim=1)
        w = torch.nn.functional.normalize(self.center, p=2, dim=1)  # noqa
        x = torch.nn.functional.normalize(emb, p=2, dim=1)  # noqa

        # TODO(exercise 2): compute scores based on w and x

        # TODO(exercise 2): calculate the score bias based on m_real and m_fake

        # TODO(exercise 2): apply alpha and softplus

        # TODO(exercise 2): returnthe final loss
        return None

    def score(self, emb: torch.Tensor):
        """
        Compute the scores based on the encoder embeddings.

        This method calculates the similarity scores between the input
        embeddings and the learned center vector. The scores can be used
        to evaluate the confidence of the model's predictions regarding
        whether the input is real or spoofed.

        Args:
            emb (torch.Tensor): Encoder embedding output of shape
                [Batch, T, enc_dim], where `Batch` is the number of samples,
                `T` is the sequence length, and `enc_dim` is the dimensionality
                of the embeddings.

        Returns:
            torch.Tensor: A tensor of shape [Batch] containing the computed
            similarity scores for each input embedding.

        Examples:
            >>> loss_fn = ASVSpoofOCSoftmaxLoss()
            >>> embeddings = torch.randn(32, 10, 128)  # 32 samples, 10 time steps
            >>> scores = loss_fn.score(embeddings)
            >>> print(scores.shape)  # Output: torch.Size([32])

        Note:
            The method normalizes both the input embeddings and the learned
            center vector before computing the scores to ensure that the
            scores are computed based on cosine similarity.

        Todo:
            - Implement the score computation logic based on the normalized
              embeddings and the center vector.
        """
        emb = torch.mean(emb, dim=1)
        w = torch.nn.functional.normalize(self.center, p=2, dim=1)  # noqa
        x = torch.nn.functional.normalize(emb, p=2, dim=1)  # noqa

        # TODO(exercise 2): compute scores
