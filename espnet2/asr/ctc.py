import logging
from typing import Optional

import torch
import torch.nn.functional as F
from typeguard import typechecked


class CTC(torch.nn.Module):
    """
    Connectionist Temporal Classification (CTC) module for sequence-to-sequence
    tasks.

    This module implements the CTC loss function, which is commonly used in
    automatic speech recognition and other sequence prediction tasks where
    the alignment between input and output sequences is unknown.

    Args:
        odim (int): Dimension of outputs (vocabulary size).
        encoder_output_size (int): Number of encoder projection units.
        dropout_rate (float, optional): Dropout rate (0.0 ~ 1.0). Default is 0.0.
        ctc_type (str, optional): Type of CTC loss to use. Options are
            "builtin", "builtin2", "gtnctc", or "brctc". Default is "builtin".
        reduce (bool, optional): Whether to reduce the CTC loss into a scalar.
            Default is True.
        ignore_nan_grad (Optional[bool], optional): If set to True, NaN gradients
            are ignored. This is kept for backward compatibility. Default is None.
        zero_infinity (bool, optional): Whether to zero infinite losses and the
            associated gradients. Default is True.
        brctc_risk_strategy (str, optional): Risk strategy for Bayes Risk CTC.
            Default is "exp".
        brctc_group_strategy (str, optional): Group strategy for Bayes Risk CTC.
            Default is "end".
        brctc_risk_factor (float, optional): Risk factor for Bayes Risk CTC.
            Default is 0.0.

    Raises:
        ValueError: If an invalid ctc_type is provided.
        ImportError: If "brctc" is selected but the K2 library is not installed.

    Examples:
        >>> ctc = CTC(odim=10, encoder_output_size=64)
        >>> hs_pad = torch.randn(32, 100, 64)  # (B, Tmax, D)
        >>> hlens = torch.randint(1, 100, (32,))  # Lengths of hidden states
        >>> ys_pad = torch.randint(0, 10, (32, 50))  # Padded target sequences
        >>> ys_lens = torch.randint(1, 50, (32,))  # Lengths of target sequences
        >>> loss = ctc(hs_pad, hlens, ys_pad, ys_lens)

    Note:
        The "builtin" and "builtin2" types use PyTorch's built-in CTC loss
        implementation, while "gtnctc" and "brctc" require additional libraries
        for their respective functionalities.

    Todo:
        - Add more detailed error handling for various input cases.
    """

    @typechecked
    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        ctc_type: str = "builtin",
        reduce: bool = True,
        ignore_nan_grad: Optional[bool] = None,
        zero_infinity: bool = True,
        brctc_risk_strategy: str = "exp",
        brctc_group_strategy: str = "end",
        brctc_risk_factor: float = 0.0,
    ):
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = torch.nn.Linear(eprojs, odim)
        self.ctc_type = ctc_type
        if ignore_nan_grad is not None:
            zero_infinity = ignore_nan_grad

        if self.ctc_type == "builtin":
            self.ctc_loss = torch.nn.CTCLoss(
                reduction="none", zero_infinity=zero_infinity
            )
        elif self.ctc_type == "builtin2":
            self.ignore_nan_grad = True
            logging.warning("builtin2")
            self.ctc_loss = torch.nn.CTCLoss(reduction="none")

        elif self.ctc_type == "gtnctc":
            from espnet.nets.pytorch_backend.gtn_ctc import GTNCTCLossFunction

            self.ctc_loss = GTNCTCLossFunction.apply

        elif self.ctc_type == "brctc":
            try:
                import k2  # noqa
            except ImportError:
                raise ImportError("You should install K2 to use Bayes Risk CTC")

            from espnet2.asr.bayes_risk_ctc import BayesRiskCTC

            self.ctc_loss = BayesRiskCTC(
                brctc_risk_strategy, brctc_group_strategy, brctc_risk_factor
            )

        else:
            raise ValueError(f'ctc_type must be "builtin" or "gtnctc": {self.ctc_type}')

        self.reduce = reduce

    def loss_fn(self, th_pred, th_target, th_ilen, th_olen) -> torch.Tensor:
        """
        Compute the CTC loss for the given predictions and targets.

        This function calculates the Connectionist Temporal Classification (CTC)
        loss between the predicted logits and the target sequences. It handles
        various types of CTC loss implementations based on the `ctc_type`
        specified during the initialization of the CTC module.

        Args:
            th_pred (torch.Tensor): The predicted logits from the model. Shape
                should be (B, L, O), where B is the batch size, L is the length
                of the sequences, and O is the number of output classes.
            th_target (torch.Tensor): The target sequences of character IDs.
                Shape should be (N,), where N is the total number of target
                characters across the batch.
            th_ilen (torch.Tensor): The lengths of the predicted sequences.
                Shape should be (B,).
            th_olen (torch.Tensor): The lengths of the target sequences.
                Shape should be (B,).

        Returns:
            torch.Tensor: The computed CTC loss value. The shape depends on the
            `reduce` attribute; if `reduce` is True, the loss is a scalar, else
            it retains the shape corresponding to the number of valid sequences.

        Raises:
            NotImplementedError: If the `ctc_type` is not recognized.
            ValueError: If `ctc_type` is neither "builtin" nor "gtnctc".

        Examples:
            >>> ctc = CTC(odim=10, encoder_output_size=5)
            >>> th_pred = torch.randn(3, 4, 10)  # Example logits for batch size 3
            >>> th_target = torch.tensor([1, 2, 3])  # Example target character IDs
            >>> th_ilen = torch.tensor([4, 4, 4])  # All sequences have length 4
            >>> th_olen = torch.tensor([1, 1, 1])  # All targets have length 1
            >>> loss = ctc.loss_fn(th_pred, th_target, th_ilen, th_olen)
            >>> print(loss)

        Note:
            The function can handle NaN gradients based on the `ignore_nan_grad`
            attribute, which can help avoid issues during training.
        """
        if self.ctc_type == "builtin" or self.ctc_type == "brctc":
            th_pred = th_pred.log_softmax(2).float()
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
            if self.ctc_type == "builtin":
                size = th_pred.size(1)
            else:
                size = loss.size(0)  # some invalid examples will be excluded

            if self.reduce:
                # Batch-size average
                loss = loss.sum() / size
            else:
                loss = loss / size
            return loss

        # builtin2 ignores nan losses using the logic below, while
        # builtin relies on the zero_infinity flag in pytorch CTC
        elif self.ctc_type == "builtin2":
            th_pred = th_pred.log_softmax(2).float()
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)

            if loss.requires_grad and self.ignore_nan_grad:
                # ctc_grad: (L, B, O)
                ctc_grad = loss.grad_fn(torch.ones_like(loss))
                ctc_grad = ctc_grad.sum([0, 2])
                indices = torch.isfinite(ctc_grad)
                size = indices.long().sum()
                if size == 0:
                    # Return as is
                    logging.warning(
                        "All samples in this mini-batch got nan grad."
                        " Returning nan value instead of CTC loss"
                    )
                elif size != th_pred.size(1):
                    logging.warning(
                        f"{th_pred.size(1) - size}/{th_pred.size(1)}"
                        " samples got nan grad."
                        " These were ignored for CTC loss."
                    )

                    # Create mask for target
                    target_mask = torch.full(
                        [th_target.size(0)],
                        1,
                        dtype=torch.bool,
                        device=th_target.device,
                    )
                    s = 0
                    for ind, le in enumerate(th_olen):
                        if not indices[ind]:
                            target_mask[s : s + le] = 0
                        s += le

                    # Calc loss again using maksed data
                    loss = self.ctc_loss(
                        th_pred[:, indices, :],
                        th_target[target_mask],
                        th_ilen[indices],
                        th_olen[indices],
                    )
            else:
                size = th_pred.size(1)

            if self.reduce:
                # Batch-size average
                loss = loss.sum() / size
            else:
                loss = loss / size
            return loss

        elif self.ctc_type == "gtnctc":
            log_probs = torch.nn.functional.log_softmax(th_pred, dim=2)
            return self.ctc_loss(log_probs, th_target, th_ilen, 0, "none")

        else:
            raise NotImplementedError

    def forward(self, hs_pad, hlens, ys_pad, ys_lens):
        """
        Calculate the Connectionist Temporal Classification (CTC) loss.

        This method computes the CTC loss given the padded hidden state
        sequences and the corresponding target sequences. It handles various
        types of CTC losses based on the `ctc_type` specified during
        initialization.

        Args:
            hs_pad (torch.Tensor): A batch of padded hidden state sequences
                with shape (B, Tmax, D), where B is the batch size,
                Tmax is the maximum sequence length, and D is the number
                of features.
            hlens (torch.Tensor): A tensor containing the lengths of the
                hidden state sequences with shape (B).
            ys_pad (torch.Tensor): A batch of padded character ID sequences
                with shape (B, Lmax), where Lmax is the maximum target
                sequence length.
            ys_lens (torch.Tensor): A tensor containing the lengths of the
                character sequences with shape (B).

        Returns:
            torch.Tensor: The computed CTC loss as a tensor. The loss is
            returned in the same device and data type as the input hidden
            states.

        Examples:
            >>> ctc = CTC(odim=10, encoder_output_size=20)
            >>> hs_pad = torch.randn(32, 50, 20)  # Example hidden states
            >>> hlens = torch.randint(1, 51, (32,))  # Example lengths
            >>> ys_pad = torch.randint(0, 10, (32, 30))  # Example targets
            >>> ys_lens = torch.randint(1, 31, (32,))  # Example target lengths
            >>> loss = ctc(hs_pad, hlens, ys_pad, ys_lens)
            >>> print(loss)

        Note:
            Ensure that the input tensors are appropriately padded and
            have the correct shapes as specified in the arguments.

        Raises:
            NotImplementedError: If the `ctc_type` is not supported.
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))

        if self.ctc_type == "brctc":
            loss = self.loss_fn(ys_hat, ys_pad, hlens, ys_lens).to(
                device=hs_pad.device, dtype=hs_pad.dtype
            )
            return loss

        elif self.ctc_type == "gtnctc":
            # gtn expects list form for ys
            ys_true = [y[y != -1] for y in ys_pad]  # parse padded ys
        else:
            # ys_hat: (B, L, D) -> (L, B, D)
            ys_hat = ys_hat.transpose(0, 1)
            # (B, L) -> (BxL,)
            ys_true = torch.cat([ys_pad[i, :l] for i, l in enumerate(ys_lens)])

        loss = self.loss_fn(ys_hat, ys_true, hlens, ys_lens).to(
            device=hs_pad.device, dtype=hs_pad.dtype
        )

        return loss

    def softmax(self, hs_pad):
        """
        Compute the softmax of frame activations.

        This method applies the softmax function to the output of the linear
        layer, converting raw logits into probabilities. It is particularly
        useful for interpreting the model's output in the context of
        classification tasks.

        Args:
            hs_pad (torch.Tensor): A 3D tensor of shape (B, Tmax, eprojs)
                where B is the batch size, Tmax is the maximum time steps,
                and eprojs is the number of encoder projection units.

        Returns:
            torch.Tensor: A 3D tensor of shape (B, Tmax, odim) containing the
            softmax probabilities for each output dimension (odim).

        Examples:
            >>> ctc = CTC(odim=10, encoder_output_size=20)
            >>> hs_pad = torch.randn(5, 15, 20)  # Example input
            >>> softmax_output = ctc.softmax(hs_pad)
            >>> print(softmax_output.shape)  # Output: torch.Size([5, 15, 10])
        """
        return F.softmax(self.ctc_lo(hs_pad), dim=2)

    def log_softmax(self, hs_pad):
        """
        Computes the log softmax of frame activations.

        This function applies the log softmax function to the output of the
        CTC layer, transforming the raw scores (logits) into log-probabilities.
        The log softmax function is particularly useful in the context of
        neural networks as it helps in numerical stability during training.

        Args:
            hs_pad (torch.Tensor): A 3D tensor of shape (B, Tmax, eprojs),
                where B is the batch size, Tmax is the maximum time steps,
                and eprojs is the number of encoder projection units.

        Returns:
            torch.Tensor: A 3D tensor of shape (B, Tmax, odim) after applying
            the log softmax function, where odim is the dimension of outputs.

        Examples:
            >>> ctc = CTC(odim=10, encoder_output_size=5)
            >>> hs_pad = torch.rand(2, 3, 5)  # Example input tensor
            >>> log_probs = ctc.log_softmax(hs_pad)
            >>> log_probs.shape
            torch.Size([2, 3, 10])

        Note:
            The output of this function can be used as input to the CTC loss
            function to compute the loss during training.
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad):
        """
        Compute the argmax of frame activations.

        This method applies the CTC linear layer to the input tensor and
        computes the argmax across the output dimension. The input should
        be a 3D tensor representing the batch of padded hidden state
        sequences.

        Args:
            hs_pad (torch.Tensor): A 3D tensor of shape (B, Tmax, eprojs),
                where B is the batch size, Tmax is the maximum sequence
                length, and eprojs is the number of encoder projection
                units.

        Returns:
            torch.Tensor: A 2D tensor of shape (B, Tmax) containing the
                indices of the maximum values along the output dimension
                for each time step.

        Examples:
            >>> ctc = CTC(odim=10, encoder_output_size=20)
            >>> hs_pad = torch.randn(4, 5, 20)  # Example input
            >>> argmax_output = ctc.argmax(hs_pad)
            >>> print(argmax_output.shape)
            torch.Size([4, 5])  # Output shape will be (B, Tmax)

        Note:
            This function is useful for decoding the predicted output
            sequences from the model after training.
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)
