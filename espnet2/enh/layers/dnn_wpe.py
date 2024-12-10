from typing import Tuple, Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import to_double, to_float
from espnet2.enh.layers.mask_estimator import MaskEstimator
from espnet2.enh.layers.wpe import wpe_one_iteration
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class DNN_WPE(torch.nn.Module):
    """
    DNN_WPE is a deep neural network-based implementation for Weighted 
Prediction Error (WPE) dereverberation.

This module utilizes a DNN mask estimator to predict the mask for 
dereverberation and applies the WPE algorithm iteratively to enhance 
the input signal.

Attributes:
    iterations (int): Number of iterations for WPE processing.
    taps (int): Number of taps used in the WPE algorithm.
    delay (int): Delay parameter for the WPE algorithm.
    eps (float): Small value to prevent division by zero.
    normalization (bool): Whether to normalize the masks.
    use_dnn_mask (bool): Flag to indicate if DNN mask estimation should be used.
    diagonal_loading (bool): Flag to indicate if diagonal loading is used.
    diag_eps (float): Small value for diagonal loading.
    mask_flooring (bool): Flag to indicate if mask flooring is applied.
    flooring_thres (float): Threshold for mask flooring.
    use_torch_solver (bool): Flag to indicate if PyTorch solver is used.

Args:
    wtype (str): Type of the network (default: "blstmp").
    widim (int): Dimension of the input features (default: 257).
    wlayers (int): Number of layers in the DNN (default: 3).
    wunits (int): Number of units in each layer (default: 300).
    wprojs (int): Number of projections (default: 320).
    dropout_rate (float): Dropout rate for the DNN (default: 0.0).
    taps (int): Number of taps for WPE (default: 5).
    delay (int): Delay parameter for WPE (default: 3).
    use_dnn_mask (bool): Whether to use DNN mask estimation (default: True).
    nmask (int): Number of masks to be predicted (default: 1).
    nonlinear (str): Nonlinearity type for DNN (default: "sigmoid").
    iterations (int): Number of iterations for WPE (default: 1).
    normalization (bool): Whether to normalize the masks (default: False).
    eps (float): Small value for numerical stability (default: 1e-6).
    diagonal_loading (bool): Whether to apply diagonal loading (default: True).
    diag_eps (float): Small value for diagonal loading (default: 1e-7).
    mask_flooring (bool): Whether to apply mask flooring (default: False).
    flooring_thres (float): Threshold for mask flooring (default: 1e-6).
    use_torch_solver (bool): Whether to use PyTorch solver (default: True).

Returns:
    Tuple[Union[torch.Tensor, ComplexTensor], torch.LongTensor, 
    Union[torch.Tensor, ComplexTensor]]:
        - enhanced: The enhanced signal (shape: (B, T, C, F)).
        - ilens: Input lengths (shape: (B,)).
        - masks: Predicted masks (shape: (B, T, C, F)).
        - power: Calculated power (shape: (B, F, T)).

Examples:
    >>> model = DNN_WPE()
    >>> input_data = torch.randn(2, 100, 1, 257)  # (B, T, C, F)
    >>> ilens = torch.tensor([100, 100])
    >>> enhanced, ilens, masks, power = model(input_data, ilens)

Note:
    This class requires PyTorch and torch_complex libraries.
    """
    def __init__(
        self,
        wtype: str = "blstmp",
        widim: int = 257,
        wlayers: int = 3,
        wunits: int = 300,
        wprojs: int = 320,
        dropout_rate: float = 0.0,
        taps: int = 5,
        delay: int = 3,
        use_dnn_mask: bool = True,
        nmask: int = 1,
        nonlinear: str = "sigmoid",
        iterations: int = 1,
        normalization: bool = False,
        eps: float = 1e-6,
        diagonal_loading: bool = True,
        diag_eps: float = 1e-7,
        mask_flooring: bool = False,
        flooring_thres: float = 1e-6,
        use_torch_solver: bool = True,
    ):
        super().__init__()
        self.iterations = iterations
        self.taps = taps
        self.delay = delay
        self.eps = eps

        self.normalization = normalization
        self.use_dnn_mask = use_dnn_mask

        self.inverse_power = True
        self.diagonal_loading = diagonal_loading
        self.diag_eps = diag_eps
        self.mask_flooring = mask_flooring
        self.flooring_thres = flooring_thres
        self.use_torch_solver = use_torch_solver

        if self.use_dnn_mask:
            self.nmask = nmask
            self.mask_est = MaskEstimator(
                wtype,
                widim,
                wlayers,
                wunits,
                wprojs,
                dropout_rate,
                nmask=nmask,
                nonlinear=nonlinear,
            )
        else:
            self.nmask = 1

    def forward(
        self, data: Union[torch.Tensor, ComplexTensor], ilens: torch.LongTensor
    ) -> Tuple[
        Union[torch.Tensor, ComplexTensor],
        torch.LongTensor,
        Union[torch.Tensor, ComplexTensor],
    ]:
        """
        DNN_WPE forward function.

    This method performs the forward pass for the DNN_WPE model, which applies
    deep neural network-based weighted prediction error (WPE) to enhance input 
    audio signals. The input can be either a standard PyTorch tensor or a 
    complex tensor, and the function returns the enhanced audio, input lengths, 
    and the calculated masks.

    Notation:
        B: Batch
        C: Channel
        T: Time or Sequence length
        F: Frequency or Some dimension of the feature vector

    Args:
        data: Input audio data of shape (B, T, C, F), where B is the batch size,
              T is the time length, C is the number of channels, and F is the 
              feature dimension.
        ilens: Input lengths of shape (B,) that indicate the valid time steps 
               for each batch element.

    Returns:
        enhanced (torch.Tensor or List[torch.Tensor]): Enhanced audio data of 
                  shape (B, T, C, F).
        ilens (torch.LongTensor): Input lengths of shape (B,) for the enhanced 
               output.
        masks (torch.Tensor or List[torch.Tensor]): Masks used in the enhancement 
               process of shape (B, T, C, F).
        power (List[torch.Tensor]): Power estimates of shape (B, F, T).

    Examples:
        >>> model = DNN_WPE()
        >>> input_data = torch.randn(2, 100, 1, 257)  # Example input
        >>> input_lengths = torch.tensor([100, 80])   # Example lengths
        >>> enhanced_output, lengths, masks, power = model(input_data, input_lengths)

    Note:
        The method performs several iterations to refine the enhanced output,
        and can apply different configurations for mask estimation, normalization, 
        and flooring based on the model's parameters.

    Raises:
        ValueError: If the input data shape does not match the expected dimensions.
        """
        # (B, T, C, F) -> (B, F, C, T)
        data = data.permute(0, 3, 2, 1)
        enhanced = [data for i in range(self.nmask)]
        masks = None
        power = None

        for i in range(self.iterations):
            # Calculate power: (..., C, T)
            power = [enh.real**2 + enh.imag**2 for enh in enhanced]
            if i == 0 and self.use_dnn_mask:
                # mask: (B, F, C, T)
                masks, _ = self.mask_est(data, ilens)
                # floor masks to increase numerical stability
                if self.mask_flooring:
                    masks = [m.clamp(min=self.flooring_thres) for m in masks]
                if self.normalization:
                    # Normalize along T
                    masks = [m / m.sum(dim=-1, keepdim=True) for m in masks]
                # (..., C, T) * (..., C, T) -> (..., C, T)
                power = [p * masks[i] for i, p in enumerate(power)]

            # Averaging along the channel axis: (..., C, T) -> (..., T)
            power = [p.mean(dim=-2).clamp(min=self.eps) for p in power]

            # enhanced: (..., C, T) -> (..., C, T)
            # NOTE(kamo): Calculate in double precision
            enhanced = [
                wpe_one_iteration(
                    to_double(data.contiguous()),
                    to_double(p),
                    taps=self.taps,
                    delay=self.delay,
                    inverse_power=self.inverse_power,
                )
                for p in power
            ]
            enhanced = [
                enh.to(dtype=data.dtype).masked_fill(make_pad_mask(ilens, enh.real), 0)
                for enh in enhanced
            ]

        # (B, F, C, T) -> (B, T, C, F)
        enhanced = [enh.permute(0, 3, 2, 1) for enh in enhanced]
        if masks is not None:
            masks = (
                [m.transpose(-1, -3) for m in masks]
                if self.nmask > 1
                else masks[0].transpose(-1, -3)
            )
        if self.nmask == 1:
            enhanced = enhanced[0]

        return enhanced, ilens, masks, power

    def predict_mask(
        self, data: Union[torch.Tensor, ComplexTensor], ilens: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Predict mask for WPE dereverberation.

This method computes the masks used in the Weighted Prediction Error (WPE)
dereverberation process. It utilizes a deep neural network (DNN) to estimate 
the masks from the input data.

Args:
    data (torch.complex64/ComplexTensor): Input tensor of shape (B, T, C, F),
        where B is the batch size, T is the time length, C is the number of
        channels, and F is the frequency dimension. The input should be in
        double precision.
    ilens (torch.Tensor): A tensor of shape (B,) representing the lengths of
        each input sequence in the batch.

Returns:
    Tuple[torch.Tensor, torch.LongTensor]: A tuple containing:
        - masks (torch.Tensor or List[torch.Tensor]): The predicted masks of 
          shape (B, T, C, F) after transposing from (B, F, C, T).
        - ilens (torch.Tensor): The unchanged lengths of each input sequence 
          in the batch, of shape (B,).

Examples:
    >>> model = DNN_WPE()
    >>> data = torch.randn(10, 100, 2, 257, dtype=torch.complex64)
    >>> ilens = torch.tensor([100] * 10)
    >>> masks, ilens = model.predict_mask(data, ilens)
    >>> print(masks.shape)  # Output: torch.Size([10, 100, 2, 257])

Note:
    This method is only available if `use_dnn_mask` is set to True during
    the initialization of the DNN_WPE model.
        """
        if self.use_dnn_mask:
            masks, ilens = self.mask_est(to_float(data.permute(0, 3, 2, 1)), ilens)
            # (B, F, C, T) -> (B, T, C, F)
            masks = [m.transpose(-1, -3) for m in masks]
            if self.nmask == 1:
                masks = masks[0]
        else:
            masks = None
        return masks, ilens
