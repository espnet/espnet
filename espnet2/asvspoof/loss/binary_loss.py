import torch

from espnet2.asvspoof.loss.abs_loss import AbsASVSpoofLoss


class ASVSpoofBinaryLoss(AbsASVSpoofLoss):
    """
    Binary loss for ASV Spoofing.

    This class implements the binary cross-entropy loss function for
    anti-spoofing tasks in speaker verification. The loss is computed
    based on the predicted probabilities of the model and the ground
    truth labels.

    Attributes:
        weight (float): A scalar value to weight the loss. Default is 1.0.

    Args:
        weight (float): A scalar value to weight the loss during training. 
            Default is 1.0.

    Methods:
        forward(pred: torch.Tensor, label: torch.Tensor, **kwargs):
            Computes the binary cross-entropy loss based on the model
            predictions and the ground truth labels.

        score(pred: torch.Tensor) -> torch.Tensor:
            Returns the predictions as is, which can be used for
            evaluation metrics.

    Examples:
        >>> loss_fn = ASVSpoofBinaryLoss(weight=0.5)
        >>> predictions = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        >>> labels = torch.tensor([[1, 0], [0, 1]])
        >>> loss = loss_fn(predictions, labels)
        >>> print(loss.item())

    Note:
        The predictions should be the raw output from the model and should
        be of shape [Batch, 2], where the second dimension represents the
        two classes (genuine and spoof). The labels should be in the same
        shape.

    Raises:
        ValueError: If the shape of predictions does not match the shape
            of labels.
    """

    def __init__(
        self,
        weight: float = 1.0,
    ):
        super().__init__()
        self.weight = weight
        self.sigmoid = torch.nn.Sigmoid()
        self.loss = torch.nn.BCELoss(reduction="mean")

    def forward(self, pred: torch.Tensor, label: torch.Tensor, **kwargs):
        """
        Compute the binary cross-entropy loss for ASV spoofing.

        This method calculates the binary cross-entropy loss between the
        predicted probabilities and the ground truth labels. The predictions
        are first passed through a sigmoid function to obtain values between
        0 and 1 before calculating the loss.

        Args:
            pred (torch.Tensor): Prediction probabilities of shape [Batch, 2],
                where each row corresponds to the predicted scores for the two
                classes (genuine and spoof).
            label (torch.Tensor): Ground truth labels of shape [Batch, 2],
                where each row contains the binary labels (0 for genuine,
                1 for spoof) corresponding to the predictions.

        Returns:
            torch.Tensor: The computed binary cross-entropy loss as a scalar
            tensor.

        Examples:
            >>> import torch
            >>> loss_fn = ASVSpoofBinaryLoss()
            >>> pred = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
            >>> label = torch.tensor([[1, 0], [0, 1]])
            >>> loss = loss_fn.forward(pred, label)
            >>> print(loss)  # Output will be a tensor representing the loss

        Note:
            The input tensors `pred` and `label` should have the same shape,
            and the values in `label` should be binary (0 or 1).

        Raises:
            ValueError: If the shapes of `pred` and `label` do not match.
        """
        loss = self.loss(self.sigmoid(pred.view(-1)), label.view(-1).float())
        return loss

    def score(self, pred: torch.Tensor):
        """
        Calculate the score from the prediction tensor.

        This method takes the prediction tensor and returns it directly as the score. 
        The score can be interpreted as the raw output probabilities from the model.

        Args:
            pred (torch.Tensor): The prediction probabilities from the model, 
                expected shape [Batch, 2].

        Returns:
            torch.Tensor: The input prediction tensor, serving as the score.

        Examples:
            >>> import torch
            >>> loss_fn = ASVSpoofBinaryLoss()
            >>> pred = torch.tensor([[0.8, 0.2], [0.4, 0.6]])
            >>> score = loss_fn.score(pred)
            >>> print(score)
            tensor([[0.8, 0.2],
                    [0.4, 0.6]])

        Note:
            The returned score is the same as the input predictions without any 
            further processing.
        """
        return pred
