import logging
from typing import Optional

import torch
import torch.nn.functional as F
from typeguard import typechecked


class CTC(torch.nn.Module):
    """
        CTC (Connectionist Temporal Classification) module.

    This class implements various CTC loss functions for sequence-to-sequence models,
    particularly useful in speech recognition tasks.

    Attributes:
        ctc_lo (torch.nn.Linear): Linear layer for CTC output.
        ctc_loss (callable): CTC loss function based on the specified type.
        dropout_rate (float): Dropout rate applied to the input.
        ctc_type (str): Type of CTC loss to use ('builtin', 'builtin2', 'gtnctc', or 'brctc').
        reduce (bool): Whether to reduce the CTC loss into a scalar.

    Args:
        odim (int): Dimension of outputs.
        encoder_output_size (int): Number of encoder projection units.
        dropout_rate (float, optional): Dropout rate (0.0 ~ 1.0). Defaults to 0.0.
        ctc_type (str, optional): Type of CTC loss. Defaults to "builtin".
        reduce (bool, optional): Whether to reduce the CTC loss. Defaults to True.
        ignore_nan_grad (bool, optional): Ignore NaN gradients (deprecated, use zero_infinity).
        zero_infinity (bool, optional): Zero infinite losses and associated gradients. Defaults to True.
        brctc_risk_strategy (str, optional): Risk strategy for Bayes Risk CTC. Defaults to "exp".
        brctc_group_strategy (str, optional): Group strategy for Bayes Risk CTC. Defaults to "end".
        brctc_risk_factor (float, optional): Risk factor for Bayes Risk CTC. Defaults to 0.0.

    Raises:
        ValueError: If ctc_type is not one of "builtin", "gtnctc", or "brctc".
        ImportError: If K2 is not installed when using Bayes Risk CTC.

    Note:
        The class supports different CTC implementations, including the built-in PyTorch CTC,
        GTN-based CTC, and Bayes Risk CTC. The choice of CTC type affects the behavior and
        performance of the loss calculation.

    Example:
        >>> ctc = CTC(odim=1000, encoder_output_size=256, ctc_type="builtin")
        >>> hs_pad = torch.randn(32, 100, 256)  # (batch_size, max_time, hidden_size)
        >>> hlens = torch.full((32,), 100)  # (batch_size,)
        >>> ys_pad = torch.randint(0, 1000, (32, 50))  # (batch_size, max_label_length)
        >>> ys_lens = torch.randint(10, 50, (32,))  # (batch_size,)
        >>> loss = ctc(hs_pad, hlens, ys_pad, ys_lens)
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
                Calculate the CTC loss based on the specified CTC type.

        This method computes the CTC loss using the predefined CTC loss function,
        which varies depending on the CTC type specified during initialization.

        Args:
            th_pred (torch.Tensor): Predicted probabilities or logits.
                Shape: (batch_size, max_time, num_classes)
            th_target (torch.Tensor): Target labels.
                Shape: (sum(target_lengths))
            th_ilen (torch.Tensor): Input lengths.
                Shape: (batch_size,)
            th_olen (torch.Tensor): Output lengths.
                Shape: (batch_size,)

        Returns:
            torch.Tensor: Computed CTC loss.

        Raises:
            NotImplementedError: If an unsupported CTC type is specified.

        Note:
            - For 'builtin' and 'brctc' types, the input is expected to be log probabilities.
            - For 'builtin2', NaN gradients are handled differently based on the 'ignore_nan_grad' flag.
            - For 'gtnctc', the input is converted to log probabilities within the method.

        Example:
            >>> ctc = CTC(odim=10, encoder_output_size=20, ctc_type="builtin")
            >>> pred = torch.randn(2, 5, 10)  # (batch_size, max_time, num_classes)
            >>> target = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])  # (sum(target_lengths))
            >>> input_length = torch.tensor([5, 5])  # (batch_size,)
            >>> target_length = torch.tensor([3, 5])  # (batch_size,)
            >>> loss = ctc.loss_fn(pred, target, input_length, target_length)
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
                Calculate CTC loss for the input sequences.

        This method applies the CTC loss calculation to the input hidden state sequences.
        It first applies a linear transformation and dropout to the input, then computes
        the CTC loss based on the specified CTC type.

        Args:
            hs_pad (torch.Tensor): Batch of padded hidden state sequences.
                Shape: (batch_size, max_time, hidden_size)
            hlens (torch.Tensor): Batch of lengths of hidden state sequences.
                Shape: (batch_size,)
            ys_pad (torch.Tensor): Batch of padded character id sequence tensor.
                Shape: (batch_size, max_label_length)
            ys_lens (torch.Tensor): Batch of lengths of character sequences.
                Shape: (batch_size,)

        Returns:
            torch.Tensor: Computed CTC loss.

        Note:
            - The method handles different CTC types ('brctc', 'gtnctc', and others) differently.
            - For 'gtnctc', the target sequences are converted to a list format.
            - For other types, the target sequences are flattened into a 1D tensor.

        Example:
            >>> ctc = CTC(odim=1000, encoder_output_size=256)
            >>> hs_pad = torch.randn(32, 100, 256)  # (batch_size, max_time, hidden_size)
            >>> hlens = torch.full((32,), 100)  # (batch_size,)
            >>> ys_pad = torch.randint(0, 1000, (32, 50))  # (batch_size, max_label_length)
            >>> ys_lens = torch.randint(10, 50, (32,))  # (batch_size,)
            >>> loss = ctc(hs_pad, hlens, ys_pad, ys_lens)
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
                Apply softmax to frame activations.

        This method applies a linear transformation followed by softmax to the input
        hidden state sequences, typically used for obtaining output probabilities.

        Args:
            hs_pad (torch.Tensor): 3D tensor of padded hidden state sequences.
                Shape: (batch_size, max_time, hidden_size)

        Returns:
            torch.Tensor: Softmax applied 3D tensor.
                Shape: (batch_size, max_time, output_dim)

        Example:
            >>> ctc = CTC(odim=1000, encoder_output_size=256)
            >>> hs_pad = torch.randn(32, 100, 256)  # (batch_size, max_time, hidden_size)
            >>> softmax_output = ctc.softmax(hs_pad)
            >>> softmax_output.shape
            torch.Size([32, 100, 1000])
        """
        return F.softmax(self.ctc_lo(hs_pad), dim=2)

    def log_softmax(self, hs_pad):
        """
                Apply log softmax to frame activations.

        This method applies a linear transformation followed by log softmax to the input
        hidden state sequences, typically used for obtaining log probabilities.

        Args:
            hs_pad (torch.Tensor): 3D tensor of padded hidden state sequences.
                Shape: (batch_size, max_time, hidden_size)

        Returns:
            torch.Tensor: Log softmax applied 3D tensor.
                Shape: (batch_size, max_time, output_dim)

        Example:
            >>> ctc = CTC(odim=1000, encoder_output_size=256)
            >>> hs_pad = torch.randn(32, 100, 256)  # (batch_size, max_time, hidden_size)
            >>> log_softmax_output = ctc.log_softmax(hs_pad)
            >>> log_softmax_output.shape
            torch.Size([32, 100, 1000])
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad):
        """
                Apply argmax to frame activations.

        This method applies a linear transformation followed by argmax to the input
        hidden state sequences, typically used for obtaining the most likely class
        for each time step.

        Args:
            hs_pad (torch.Tensor): 3D tensor of padded hidden state sequences.
                Shape: (batch_size, max_time, hidden_size)

        Returns:
            torch.Tensor: Argmax applied 2D tensor.
                Shape: (batch_size, max_time)

        Example:
            >>> ctc = CTC(odim=1000, encoder_output_size=256)
            >>> hs_pad = torch.randn(32, 100, 256)  # (batch_size, max_time, hidden_size)
            >>> argmax_output = ctc.argmax(hs_pad)
            >>> argmax_output.shape
            torch.Size([32, 100])
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)
