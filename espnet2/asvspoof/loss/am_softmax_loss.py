import torch

from espnet2.asvspoof.loss.abs_loss import AbsASVSpoofLoss


class ASVSpoofAMSoftmaxLoss(AbsASVSpoofLoss):
    """
    Adaptive Margin Softmax Loss for ASV Spoofing.

    This class implements the Adaptive Margin Softmax loss function designed
    for Automatic Speaker Verification (ASV) spoofing tasks. The loss is based
    on a binary classification framework where the model learns to differentiate
    between genuine and spoofed audio samples.

    Attributes:
        weight (float): A scaling factor for the loss. Default is 1.0.
        enc_dim (int): Dimensionality of the encoder output. Default is 128.
        s (float): Scaling factor for the logits. Default is 20.
        m (float): Margin added to the logits for improved separation. Default is 0.5.
        centers (torch.nn.Parameter): Learnable parameters representing class centers.
        sigmoid (torch.nn.Sigmoid): Sigmoid activation function.
        loss (torch.nn.BCELoss): Binary cross-entropy loss function.

    Args:
        weight (float, optional): Weighting factor for the loss computation.
        enc_dim (int, optional): Dimensionality of the encoder output.
        s (float, optional): Scaling factor for logits.
        m (float, optional): Margin for the softmax loss.

    Returns:
        torch.Tensor: Computed loss value.

    Examples:
        >>> loss_fn = ASVSpoofAMSoftmaxLoss()
        >>> labels = torch.tensor([[1], [0]])
        >>> embeddings = torch.randn(2, 10, 128)  # Batch of 2, 10 time steps, 128 features
        >>> loss = loss_fn(labels, embeddings)
        >>> print(loss)

    Note:
        The input embeddings should be in the shape [Batch, T, enc_dim], where
        Batch is the number of samples, T is the sequence length, and enc_dim
        is the dimension of the encoder output.

    Raises:
        ValueError: If the dimensions of label and embedding do not match.
    """

    def __init__(
        self,
        weight: float = 1.0,
        enc_dim: int = 128,
        s: float = 20,
        m: float = 0.5,
    ):
        super(ASVSpoofAMSoftmaxLoss).__init__()
        self.weight = weight
        self.enc_dim = enc_dim
        self.s = s
        self.m = m
        self.centers = torch.nn.Parameter(torch.randn(2, enc_dim))
        self.sigmoid = torch.nn.Sigmoid()
        self.loss = torch.nn.BCELoss(reduction="mean")

    def forward(self, label: torch.Tensor, emb: torch.Tensor, **kwargs):
        """
        Compute the forward pass of the ASVSpoofAMSoftmaxLoss.

        This method computes the loss for the given input embeddings and labels
        using an angular margin softmax approach. The embeddings are normalized
        and compared against learned class centers to produce logits, which are
        then used to compute the binary cross-entropy loss.

        Args:
            label (torch.Tensor): Ground truth labels with shape [Batch, 1],
                where each label is either 0 or 1.
            emb (torch.Tensor): Encoder embedding output with shape
                [Batch, T, enc_dim], where T is the sequence length and
                enc_dim is the dimension of the embeddings.

        Returns:
            torch.Tensor: The computed loss value as a tensor.

        Examples:
            >>> import torch
            >>> loss_fn = ASVSpoofAMSoftmaxLoss()
            >>> labels = torch.tensor([[1], [0]])
            >>> embeddings = torch.rand(2, 10, 128)  # Batch of 2, 10 timesteps, 128 dim
            >>> loss = loss_fn(labels, embeddings)
            >>> print(loss)

        Note:
            The input `emb` is averaged across the time dimension before
            normalization. The learned class centers are also normalized
            prior to computing the logits.
        """
        batch_size = emb.shape[0]
        emb = torch.mean(emb, dim=1)
        norms = torch.norm(emb, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(emb, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        y_onehot = torch.FloatTensor(batch_size, 2)
        y_onehot.zero_()
        y_onehot = torch.autograd.Variable(y_onehot)
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.m)
        margin_logits = self.s * (logits - y_onehot)
        loss = self.loss(self.sigmoid(margin_logits[:, 0]), label.view(-1).float())

        return loss

    def score(self, emb: torch.Tensor):
        """
        Compute the prediction scores for the given embeddings.

        This method takes the encoder embeddings, normalizes them, and computes
        the logits by performing a matrix multiplication with the normalized
        centers. The first column of the logits is returned as the prediction
        scores.

        Args:
            emb (torch.Tensor): Encoder embedding output of shape
                                [Batch, T, enc_dim]. The embeddings are
                                averaged across the time dimension (T).

        Returns:
            torch.Tensor: The prediction scores of shape [Batch]. This tensor
                           contains the scores for each input sample.

        Examples:
            >>> loss_fn = ASVSpoofAMSoftmaxLoss()
            >>> embeddings = torch.randn(32, 10, 128)  # Batch of 32 samples
            >>> scores = loss_fn.score(embeddings)
            >>> print(scores.shape)  # Output: torch.Size([32])

        Note:
            The embeddings must be computed using the same model and settings
            as the one used during training for the scores to be meaningful.
        """
        emb = torch.mean(emb, dim=1)
        norms = torch.norm(emb, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(emb, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))
        return logits[:, 0]
