# Copyright 2022 Hitachi LTD. (Nelson Yalta)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""ProDiff related loss module for ESPnet2."""

from math import exp
from typing import Tuple

import torch
from torch.nn import functional as F
from typeguard import typechecked

from espnet.nets.pytorch_backend.fastspeech.duration_predictor import (  # noqa: H301
    DurationPredictorLoss,
)
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """
        Gaussian Noise generation function.

    This function generates a Gaussian noise tensor based on the specified window
    size and sigma. The resulting tensor can be used for various applications
    such as data augmentation or simulating noise in signal processing tasks.

    Args:
        window_size (int): The size of the window for the Gaussian function.
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: A normalized tensor representing Gaussian noise.

    Examples:
        >>> noise = gaussian(window_size=5, sigma=1.0)
        >>> print(noise)
        tensor([0.0585, 0.2419, 0.3879, 0.2419, 0.0585])

    Note:
        The generated Gaussian noise is normalized such that the sum of the
        tensor elements equals 1.
    """
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


class SSimLoss(torch.nn.Module):
    """
        SSimLoss.

    This is an implementation of structural similarity (SSIM) loss. The SSIM
    loss measures the similarity between two images, with a focus on luminance,
    contrast, and structure, which is often used in image processing tasks.

    This code is modified from https://github.com/Po-Hsun-Su/pytorch-ssim.

    Attributes:
        bias (float): Value of the bias added to the outputs and target.
        win_len (int): Size of the SSIM window.
        channels (int): Number of channels in the input tensors.
        average (bool): Flag to determine if the loss should be averaged.

    Args:
        bias (float, optional): Value of the bias. Defaults to 6.0.
        window_size (int, optional): Window size for SSIM calculation. Defaults to 11.
        channels (int, optional): Number of channels in the input. Defaults to 1.
        reduction (str, optional): Type of reduction during the loss calculation.
            Can be "none", "mean". Defaults to "none".

    Returns:
        Tensor: Loss scalar value.

    Examples:
        >>> ssim_loss = SSimLoss()
        >>> output = torch.randn(1, 1, 256, 256)
        >>> target = torch.randn(1, 1, 256, 256)
        >>> loss = ssim_loss(output, target)
        >>> print(loss)

    Raises:
        ValueError: If the input tensors do not have the same shape.
    """

    def __init__(
        self,
        bias: float = 6.0,
        window_size: int = 11,
        channels: int = 1,
        reduction: str = "none",
    ):
        """Initialization.

        Args:
            bias (float, optional): value of the bias. Defaults to 6.0.
            window_size (int, optional): Window size. Defaults to 11.
            channels (int, optional): Number of channels. Defaults to 1.
            reduction (str, optional): Type of reduction during the loss
                calculation. Defaults to "none".

        """
        super().__init__()
        self.bias = bias
        self.win_len = window_size
        self.channels = channels
        self.average = False
        if reduction == "mean":
            self.average = True

        win1d = gaussian(window_size, 1.5).unsqueeze(1)
        win2d = win1d.mm(win1d.t()).float().unsqueeze(0).unsqueeze(0)
        self.window = torch.Tensor(
            win2d.expand(channels, 1, window_size, window_size).contiguous()
        )

    def forward(self, outputs: torch.Tensor, target: torch.Tensor):
        """
                Calculate forward propagation.

        This method computes the loss values for the ProDiff loss function. It takes
        various outputs from the model, as well as the target values, and computes
        the L1 loss, duration predictor loss, pitch predictor loss, and energy
        predictor loss. The method can also apply masking to ignore padded parts of
        the sequences if specified.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, T_feats, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, T_feats, odim).
            d_outs (LongTensor): Batch of outputs of duration predictor (B, T_text).
            p_outs (Tensor): Batch of outputs of pitch predictor (B, T_text, 1).
            e_outs (Tensor): Batch of outputs of energy predictor (B, T_text, 1).
            ys (Tensor): Batch of target features (B, T_feats, odim).
            ds (LongTensor): Batch of durations (B, T_text).
            ps (Tensor): Batch of target token-averaged pitch (B, T_text, 1).
            es (Tensor): Batch of target token-averaged energy (B, T_text, 1).
            ilens (LongTensor): Batch of the lengths of each input (B,).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
                - L1 loss value.
                - Duration predictor loss value.
                - Pitch predictor loss value.
                - Energy predictor loss value.

        Examples:
            >>> after_outs = torch.rand(2, 100, 80)
            >>> before_outs = torch.rand(2, 100, 80)
            >>> d_outs = torch.randint(1, 10, (2, 20))
            >>> p_outs = torch.rand(2, 20, 1)
            >>> e_outs = torch.rand(2, 20, 1)
            >>> ys = torch.rand(2, 100, 80)
            >>> ds = torch.randint(1, 10, (2, 20))
            >>> ps = torch.rand(2, 20, 1)
            >>> es = torch.rand(2, 20, 1)
            >>> ilens = torch.tensor([100, 90])
            >>> olens = torch.tensor([100, 90])
            >>> l1_loss, ssim_loss, duration_loss, pitch_loss, energy_loss = loss_module.forward(
            ...     after_outs, before_outs, d_outs, p_outs, e_outs, ys, ds, ps, es, ilens, olens
            ... )

        Note:
            This method assumes that the inputs are properly shaped and on the same
            device.
        """
        with torch.no_grad():
            dim = target.size(-1)
            mask = target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)
        outputs = outputs.unsqueeze(1) + self.bias
        target = target.unsqueeze(1) + self.bias
        loss = 1 - self.ssim(outputs, target)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def ssim(self, tensor1: torch.Tensor, tensor2: torch.Tensor):
        """
                SSimLoss.

        This class implements the structural similarity (SSIM) loss function, which
        measures the similarity between two images. This code is modified from
        https://github.com/Po-Hsun-Su/pytorch-ssim.

        Attributes:
            bias (float): The bias value added to the outputs and targets.
            win_len (int): The size of the Gaussian window used in SSIM calculation.
            channels (int): The number of channels in the input tensors.
            average (bool): Indicates whether to average the SSIM values.

        Args:
            bias (float, optional): Value of the bias. Defaults to 6.0.
            window_size (int, optional): Size of the Gaussian window. Defaults to 11.
            channels (int, optional): Number of channels. Defaults to 1.
            reduction (str, optional): Type of reduction during the loss calculation.
                Defaults to "none".

        Returns:
            Tensor: The calculated SSIM loss.

        Examples:
            >>> ssim_loss = SSimLoss()
            >>> output = torch.randn(1, 1, 256, 256)
            >>> target = torch.randn(1, 1, 256, 256)
            >>> loss = ssim_loss(output, target)
            >>> print(loss)

        Raises:
            ValueError: If the input tensors do not have the same shape.

        Note:
            This loss is particularly useful for tasks involving image quality
            assessment, such as image denoising and super-resolution.
        """
        window = self.window.to(tensor1.device)
        mu1 = F.conv2d(tensor1, window, padding=self.win_len // 2, groups=self.channels)
        mu2 = F.conv2d(tensor2, window, padding=self.win_len // 2, groups=self.channels)
        mu_corr = mu1 * mu2

        mu1 = mu1.pow(2)
        mu2 = mu2.pow(2)

        sigma1 = (
            F.conv2d(
                tensor1 * tensor1,
                window,
                padding=self.win_len // 2,
                groups=self.channels,
            )
            - mu1
        )

        sigma2 = (
            F.conv2d(
                tensor2 * tensor2,
                window,
                padding=self.win_len // 2,
                groups=self.channels,
            )
            - mu2
        )

        sigma_corr = (
            F.conv2d(
                tensor1 * tensor2,
                window,
                padding=self.win_len // 2,
                groups=self.channels,
            )
            - mu_corr
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu_corr + C1) * (2 * sigma_corr + C2)) / (
            (mu1 + mu2 + C1) * (sigma1 + sigma2 + C2)
        )
        if self.average:
            return ssim_map.mean()
        return ssim_map.mean(1)


class ProDiffLoss(torch.nn.Module):
    """
        Loss function module for ProDiffLoss.

    This module implements the ProDiffLoss, which is used for training models
    in the ESPnet2 text-to-speech framework. It combines multiple loss criteria
    to optimize the performance of the model, including L1 loss, SSIM loss,
    duration prediction loss, pitch prediction loss, and energy prediction loss.

    Attributes:
        use_masking (bool): Whether to apply masking for padded parts in loss
            calculation.
        use_weighted_masking (bool): Whether to apply weighted masking in loss
            calculation.

    Args:
        use_masking (bool): Whether to apply masking for padded part in loss
            calculation.
        use_weighted_masking (bool): Whether to apply weighted masking in loss
            calculation.

    Raises:
        AssertionError: If both use_masking and use_weighted_masking are True.

    Examples:
        >>> loss_fn = ProDiffLoss(use_masking=True, use_weighted_masking=False)
        >>> l1_loss, ssim_loss, duration_loss, pitch_loss, energy_loss = loss_fn(
        ...     after_outs, before_outs, d_outs, p_outs, e_outs, ys, ds, ps, es,
        ...     ilens, olens)
    """

    @typechecked
    def __init__(
        self,
        use_masking: bool = True,
        use_weighted_masking: bool = False,
    ):
        """Initialize feed-forward Transformer loss module.

        Args:
            use_masking (bool): Whether to apply masking for padded part in loss
                calculation.
            use_weighted_masking (bool): Whether to weighted masking in loss
                calculation.

        """
        super().__init__()

        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)
        self.ssim_criterion = SSimLoss(reduction="none")

    def forward(
        self,
        after_outs: torch.Tensor,
        before_outs: torch.Tensor,
        d_outs: torch.Tensor,
        p_outs: torch.Tensor,
        e_outs: torch.Tensor,
        ys: torch.Tensor,
        ds: torch.Tensor,
        ps: torch.Tensor,
        es: torch.Tensor,
        ilens: torch.Tensor,
        olens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
                Calculate forward propagation.

        This method computes the loss values based on the outputs from the model and
        the corresponding target values. It utilizes various loss functions such as
        L1 loss, SSIM loss, and others, depending on the specified configuration
        for masking and weighted masking.

        Args:
            after_outs (torch.Tensor): Batch of outputs after postnets (B, T_feats, odim).
            before_outs (torch.Tensor): Batch of outputs before postnets (B, T_feats, odim).
            d_outs (torch.LongTensor): Batch of outputs of duration predictor (B, T_text).
            p_outs (torch.Tensor): Batch of outputs of pitch predictor (B, T_text, 1).
            e_outs (torch.Tensor): Batch of outputs of energy predictor (B, T_text, 1).
            ys (torch.Tensor): Batch of target features (B, T_feats, odim).
            ds (torch.LongTensor): Batch of durations (B, T_text).
            ps (torch.Tensor): Batch of target token-averaged pitch (B, T_text, 1).
            es (torch.Tensor): Batch of target token-averaged energy (B, T_text, 1).
            ilens (torch.LongTensor): Batch of the lengths of each input (B,).
            olens (torch.LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - L1 loss value.
                - SSIM loss value.
                - Duration predictor loss value.
                - Pitch predictor loss value.
                - Energy predictor loss value.

        Examples:
            # Example usage of the forward method
            l1_loss, ssim_loss, duration_loss, pitch_loss, energy_loss =
            loss_module.forward(after_outs, before_outs, d_outs, p_outs, e_outs,
                                 ys, ds, ps, es, ilens, olens)

        Note:
            The method can apply masking for padded parts based on the
            configuration specified during the initialization of the class.

        Todo:
            - Implement additional loss functions if required.
        """
        # First SSIM before masks
        ssim_loss = self.ssim_criterion(before_outs, ys)

        # apply mask to remove padded part
        if self.use_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            before_outs = before_outs.masked_select(out_masks)
            if after_outs is not None:
                after_outs = after_outs.masked_select(out_masks)
            ys = ys.masked_select(out_masks)
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            d_outs = d_outs.masked_select(duration_masks)
            ds = ds.masked_select(duration_masks)
            pitch_masks = make_non_pad_mask(ilens).unsqueeze(-1).to(ys.device)
            p_outs = p_outs.masked_select(pitch_masks)
            e_outs = e_outs.masked_select(pitch_masks)
            ps = ps.masked_select(pitch_masks)
            es = es.masked_select(pitch_masks)

        # calculate loss
        l1_loss = self.l1_criterion(before_outs, ys)
        if after_outs is not None:
            l1_loss += self.l1_criterion(after_outs, ys)

        duration_loss = self.duration_criterion(d_outs, ds)
        pitch_loss = self.mse_criterion(p_outs, ps)
        energy_loss = self.mse_criterion(e_outs, es)

        # make weighted mask and apply it
        if self.use_weighted_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
            out_weights /= ys.size(0) * ys.size(2)
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            duration_weights = (
                duration_masks.float() / duration_masks.sum(dim=1, keepdim=True).float()
            )
            duration_weights /= ds.size(0)

            # apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(out_masks).sum()
            duration_loss = (
                duration_loss.mul(duration_weights).masked_select(duration_masks).sum()
            )
            pitch_masks = duration_masks.unsqueeze(-1)
            pitch_weights = duration_weights.unsqueeze(-1)
            pitch_loss = pitch_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
            energy_loss = (
                energy_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
            )

        return l1_loss, ssim_loss, duration_loss, pitch_loss, energy_loss
