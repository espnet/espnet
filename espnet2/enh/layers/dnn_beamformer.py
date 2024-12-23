"""DNN beamformer module."""

import logging
from typing import List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from torch.nn import functional as F
from torch_complex.tensor import ComplexTensor

import espnet2.enh.layers.beamformer as bf_v1
import espnet2.enh.layers.beamformer_th as bf_v2
from espnet2.enh.layers.complex_utils import stack, to_double, to_float
from espnet2.enh.layers.mask_estimator import MaskEstimator

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")
is_torch_1_12_1_plus = V(torch.__version__) >= V("1.12.1")


BEAMFORMER_TYPES = (
    # Minimum Variance Distortionless Response beamformer
    "mvdr",  # RTF-based formula
    "mvdr_souden",  # Souden's solution
    # Minimum Power Distortionless Response beamformer
    "mpdr",  # RTF-based formula
    "mpdr_souden",  # Souden's solution
    # weighted MPDR beamformer
    "wmpdr",  # RTF-based formula
    "wmpdr_souden",  # Souden's solution
    # Weighted Power minimization Distortionless response beamformer
    "wpd",  # RTF-based formula
    "wpd_souden",  # Souden's solution
    # Multi-channel Wiener Filter (MWF) and weighted MWF
    "mwf",
    "wmwf",
    # Speech Distortion Weighted (SDW) MWF
    "sdw_mwf",
    # Rank-1 MWF
    "r1mwf",
    # Linearly Constrained Minimum Variance beamformer
    "lcmv",
    # Linearly Constrained Minimum Power beamformer
    "lcmp",
    # weighted Linearly Constrained Minimum Power beamformer
    "wlcmp",
    # Generalized Eigenvalue beamformer
    "gev",
    "gev_ban",  # with blind analytic normalization (BAN) post-filtering
    # time-frequency-bin-wise switching (TFS) MVDR beamformer
    "mvdr_tfs",
    "mvdr_tfs_souden",
)


class DNN_Beamformer(torch.nn.Module):
    """
        DNN mask based Beamformer.

    This class implements a deep neural network (DNN) based beamformer for
    enhancing speech signals in multi-channel audio. The beamformer utilizes
    various algorithms, including MVDR, MPDR, and WPD, and is capable of
    estimating masks for speech and noise.

    Citation:
        Multichannel End-to-end Speech Recognition; T. Ochiai et al., 2017;
        http://proceedings.mlr.press/v70/ochiai17a/ochiai17a.pdf

    Attributes:
        mask (MaskEstimator): An instance of the MaskEstimator used for
            estimating masks for beamforming.
        ref (AttentionReference or None): An optional attention-based reference
            used for beamforming.
        ref_channel (int): Index of the reference channel for beamforming.
        use_noise_mask (bool): Flag indicating whether to use noise mask.
        num_spk (int): Number of speakers to separate.
        nmask (int): Number of masks to be estimated.
        beamformer_type (str): Type of beamformer to use (e.g., "mvdr_souden").
        rtf_iterations (int): Number of iterations for estimating the RTF.
        mwf_mu (float): Weight for noise suppression in SDW-MWF.
        btaps (int): Number of taps for WPD beamformer.
        bdelay (int): Delay for WPD beamformer.
        eps (float): Small value to avoid division by zero.
        diagonal_loading (bool): Flag for applying diagonal loading.
        diag_eps (float): Small value for diagonal loading.
        mask_flooring (bool): Flag for applying mask flooring.
        flooring_thres (float): Threshold for mask flooring.
        use_torch_solver (bool): Flag indicating whether to use Torch solver.

    Args:
        bidim (int): Input feature dimension.
        btype (str): Type of DNN architecture (default: "blstmp").
        blayers (int): Number of layers in the DNN (default: 3).
        bunits (int): Number of units in each DNN layer (default: 300).
        bprojs (int): Number of projections (default: 320).
        num_spk (int): Number of speakers (default: 1).
        use_noise_mask (bool): Whether to use noise mask (default: True).
        nonlinear (str): Nonlinear activation function (default: "sigmoid").
        dropout_rate (float): Dropout rate (default: 0.0).
        badim (int): Dimension for attention reference (default: 320).
        ref_channel (int): Index of reference channel (default: -1).
        beamformer_type (str): Type of beamformer (default: "mvdr_souden").
        rtf_iterations (int): Number of iterations for RTF estimation (default: 2).
        mwf_mu (float): Noise suppression weight for SDW-MWF (default: 1.0).
        eps (float): Small constant to prevent division by zero (default: 1e-6).
        diagonal_loading (bool): Flag for diagonal loading (default: True).
        diag_eps (float): Small value for diagonal loading (default: 1e-7).
        mask_flooring (bool): Flag for applying mask flooring (default: False).
        flooring_thres (float): Threshold for mask flooring (default: 1e-6).
        use_torch_solver (bool): Use Torch solver (default: True).
        use_torchaudio_api (bool): Use torchaudio API (default: False).
        btaps (int): Number of taps for WPD (default: 5).
        bdelay (int): Delay for WPD (default: 3).

    Examples:
        # Initialize the DNN_Beamformer
        beamformer = DNN_Beamformer(bidim=128, num_spk=2)

        # Forward pass through the beamformer
        enhanced, ilens, masks = beamformer(data, ilens)

    Raises:
        ValueError: If an unsupported beamformer type is provided or if
            the number of speakers is less than 1.
    """

    def __init__(
        self,
        bidim,
        btype: str = "blstmp",
        blayers: int = 3,
        bunits: int = 300,
        bprojs: int = 320,
        num_spk: int = 1,
        use_noise_mask: bool = True,
        nonlinear: str = "sigmoid",
        dropout_rate: float = 0.0,
        badim: int = 320,
        ref_channel: int = -1,
        beamformer_type: str = "mvdr_souden",
        rtf_iterations: int = 2,
        mwf_mu: float = 1.0,
        eps: float = 1e-6,
        diagonal_loading: bool = True,
        diag_eps: float = 1e-7,
        mask_flooring: bool = False,
        flooring_thres: float = 1e-6,
        use_torch_solver: bool = True,
        # False to use old APIs; True to use torchaudio-based new APIs
        use_torchaudio_api: bool = False,
        # only for WPD beamformer
        btaps: int = 5,
        bdelay: int = 3,
    ):
        super().__init__()
        bnmask = num_spk + 1 if use_noise_mask else num_spk
        self.mask = MaskEstimator(
            btype,
            bidim,
            blayers,
            bunits,
            bprojs,
            dropout_rate,
            nmask=bnmask,
            nonlinear=nonlinear,
        )
        self.ref = (
            AttentionReference(bidim, badim, eps=eps) if ref_channel < 0 else None
        )
        self.ref_channel = ref_channel

        self.use_noise_mask = use_noise_mask
        assert num_spk >= 1, num_spk
        self.num_spk = num_spk
        self.nmask = bnmask

        if beamformer_type not in BEAMFORMER_TYPES:
            raise ValueError("Not supporting beamformer_type=%s" % beamformer_type)
        if (
            beamformer_type == "mvdr_souden" or not beamformer_type.endswith("_souden")
        ) and not use_noise_mask:
            if num_spk == 1:
                logging.warning(
                    "Initializing %s beamformer without noise mask "
                    "estimator (single-speaker case)" % beamformer_type.upper()
                )
                logging.warning(
                    "(1 - speech_mask) will be used for estimating noise "
                    "PSD in %s beamformer!" % beamformer_type.upper()
                )
            else:
                logging.warning(
                    "Initializing %s beamformer without noise mask "
                    "estimator (multi-speaker case)" % beamformer_type.upper()
                )
                logging.warning(
                    "Interference speech masks will be used for estimating "
                    "noise PSD in %s beamformer!" % beamformer_type.upper()
                )

        self.beamformer_type = beamformer_type
        if not beamformer_type.endswith("_souden"):
            assert rtf_iterations >= 2, rtf_iterations
        # number of iterations in power method for estimating the RTF
        self.rtf_iterations = rtf_iterations
        # noise suppression weight in SDW-MWF
        self.mwf_mu = mwf_mu

        assert btaps >= 0 and bdelay >= 0, (btaps, bdelay)
        self.btaps = btaps
        self.bdelay = bdelay if self.btaps > 0 else 1
        self.eps = eps
        self.diagonal_loading = diagonal_loading
        self.diag_eps = diag_eps
        self.mask_flooring = mask_flooring
        self.flooring_thres = flooring_thres
        self.use_torch_solver = use_torch_solver
        if not use_torch_solver:
            logging.warning(
                "The `use_torch_solver` argument has been deprecated. "
                "Now it will always be true in DNN_Beamformer"
            )

        if use_torchaudio_api and is_torch_1_12_1_plus:
            self.bf_func = bf_v2
        else:
            self.bf_func = bf_v1

    def forward(
        self,
        data: Union[torch.Tensor, ComplexTensor],
        ilens: torch.LongTensor,
        powers: Optional[List[torch.Tensor]] = None,
        oracle_masks: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[Union[torch.Tensor, ComplexTensor], torch.LongTensor, torch.Tensor]:
        """
            DNN_Beamformer forward function.

        This method performs the forward pass for the DNN-based beamformer,
        applying the beamforming process to the input data. It takes in the
        data, input lengths, optional power spectra, and oracle masks, and
        produces enhanced signals, updated input lengths, and estimated masks.

        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Frequency

        Args:
            data (torch.complex64/ComplexTensor): Input tensor of shape (B, T, C, F).
            ilens (torch.Tensor): Input lengths of shape (B,).
            powers (List[torch.Tensor] or None): Optional power spectra used for
                wMPDR or WPD with shape (B, F, T).
            oracle_masks (List[torch.Tensor] or None): Optional oracle masks of
                shape (B, F, C, T). If provided, these masks will be used instead
                of the computed masks.

        Returns:
            enhanced (torch.complex64/ComplexTensor): Enhanced output of shape (B, T, F).
            ilens (torch.Tensor): Updated input lengths of shape (B,).
            masks (torch.Tensor): Estimated masks of shape (B, T, C, F).

        Examples:
            >>> data = torch.randn(4, 160, 2, 64, dtype=torch.complex64)
            >>> ilens = torch.tensor([160, 160, 160, 160])
            >>> enhanced, ilens, masks = model.forward(data, ilens)

        Note:
            The forward method assumes that the input data is complex and that the
            beamforming statistics are properly initialized in the DNN_Beamformer
            class. If oracle masks are provided, they should have the correct shape
            to match the number of channels in the input data.

        Raises:
            ValueError: If the specified beamformer type is not supported.
        """
        # data (B, T, C, F) -> (B, F, C, T)
        data = data.permute(0, 3, 2, 1)
        data_d = to_double(data)

        # mask: [(B, F, C, T)]
        if oracle_masks is not None:
            masks = oracle_masks
        else:
            masks, _ = self.mask(data, ilens)
        assert self.nmask == len(masks), len(masks)
        # floor masks to increase numerical stability
        if self.mask_flooring:
            masks = [torch.clamp(m, min=self.flooring_thres) for m in masks]

        if self.num_spk == 1:  # single-speaker case
            if self.use_noise_mask:
                # (mask_speech, mask_noise)
                mask_speech, mask_noise = masks
            else:
                # (mask_speech,)
                mask_speech = masks[0]
                mask_noise = 1 - mask_speech

            if self.beamformer_type in ("lcmv", "lcmp", "wlcmp"):
                raise NotImplementedError("Single source is not supported yet")
            beamformer_stats = self.bf_func.prepare_beamformer_stats(
                data_d,
                [mask_speech],
                mask_noise,
                powers=powers,
                beamformer_type=self.beamformer_type,
                bdelay=self.bdelay,
                btaps=self.btaps,
                eps=self.eps,
            )

            if self.beamformer_type in ("mvdr", "mpdr", "wmpdr", "wpd"):
                enhanced, ws = self.apply_beamforming(
                    data,
                    ilens,
                    beamformer_stats["psd_n"],
                    beamformer_stats["psd_speech"],
                    psd_distortion=beamformer_stats["psd_distortion"],
                )
            elif (
                self.beamformer_type.endswith("_souden")
                or self.beamformer_type == "mwf"
                or self.beamformer_type == "wmwf"
                or self.beamformer_type == "sdw_mwf"
                or self.beamformer_type == "r1mwf"
                or self.beamformer_type.startswith("gev")
            ):
                enhanced, ws = self.apply_beamforming(
                    data,
                    ilens,
                    beamformer_stats["psd_n"],
                    beamformer_stats["psd_speech"],
                )
            else:
                raise ValueError(
                    "Not supporting beamformer_type={}".format(self.beamformer_type)
                )

            # (..., F, T) -> (..., T, F)
            enhanced = enhanced.transpose(-1, -2)
        else:  # multi-speaker case
            if self.use_noise_mask:
                # (mask_speech1, ..., mask_noise)
                mask_speech = list(masks[:-1])
                mask_noise = masks[-1]
            else:
                # (mask_speech1, ..., mask_speechX)
                mask_speech = list(masks)
                mask_noise = None

            beamformer_stats = self.bf_func.prepare_beamformer_stats(
                data_d,
                mask_speech,
                mask_noise,
                powers=powers,
                beamformer_type=self.beamformer_type,
                bdelay=self.bdelay,
                btaps=self.btaps,
                eps=self.eps,
            )
            if self.beamformer_type in ("lcmv", "lcmp", "wlcmp"):
                rtf_mat = self.bf_func.get_rtf_matrix(
                    beamformer_stats["psd_speech"],
                    beamformer_stats["psd_distortion"],
                    diagonal_loading=self.diagonal_loading,
                    ref_channel=self.ref_channel,
                    rtf_iterations=self.rtf_iterations,
                    diag_eps=self.diag_eps,
                )

            enhanced, ws = [], []
            for i in range(self.num_spk):
                # treat all other speakers' psd_speech as noises
                if self.beamformer_type in ("mvdr", "mvdr_tfs", "wmpdr", "wpd"):
                    enh, w = self.apply_beamforming(
                        data,
                        ilens,
                        beamformer_stats["psd_n"][i],
                        beamformer_stats["psd_speech"][i],
                        psd_distortion=beamformer_stats["psd_distortion"][i],
                    )
                elif self.beamformer_type in (
                    "mvdr_souden",
                    "mvdr_tfs_souden",
                    "wmpdr_souden",
                    "wpd_souden",
                    "wmwf",
                    "sdw_mwf",
                    "r1mwf",
                    "gev",
                    "gev_ban",
                ):
                    enh, w = self.apply_beamforming(
                        data,
                        ilens,
                        beamformer_stats["psd_n"][i],
                        beamformer_stats["psd_speech"][i],
                    )
                elif self.beamformer_type == "mpdr":
                    enh, w = self.apply_beamforming(
                        data,
                        ilens,
                        beamformer_stats["psd_n"],
                        beamformer_stats["psd_speech"][i],
                        psd_distortion=beamformer_stats["psd_distortion"][i],
                    )
                elif self.beamformer_type in ("mpdr_souden", "mwf"):
                    enh, w = self.apply_beamforming(
                        data,
                        ilens,
                        beamformer_stats["psd_n"],
                        beamformer_stats["psd_speech"][i],
                    )
                elif self.beamformer_type == "lcmp":
                    enh, w = self.apply_beamforming(
                        data,
                        ilens,
                        beamformer_stats["psd_n"],
                        beamformer_stats["psd_speech"][i],
                        rtf_mat=rtf_mat,
                        spk=i,
                    )
                elif self.beamformer_type in ("lcmv", "wlcmp"):
                    enh, w = self.apply_beamforming(
                        data,
                        ilens,
                        beamformer_stats["psd_n"][i],
                        beamformer_stats["psd_speech"][i],
                        rtf_mat=rtf_mat,
                        spk=i,
                    )
                else:
                    raise ValueError(
                        "Not supporting beamformer_type={}".format(self.beamformer_type)
                    )

                # (..., F, T) -> (..., T, F)
                enh = enh.transpose(-1, -2)
                enhanced.append(enh)
                ws.append(w)

        # (..., F, C, T) -> (..., T, C, F)
        masks = [m.transpose(-1, -3) for m in masks]
        return enhanced, ilens, masks

    def apply_beamforming(
        self,
        data,
        ilens,
        psd_n,
        psd_speech,
        psd_distortion=None,
        rtf_mat=None,
        spk=0,
    ):
        """
            Beamforming with the provided statistics.

        This method applies beamforming techniques using the provided noise and
        speech covariance matrices, as well as other optional parameters. The
        implementation varies based on the type of beamformer being used, such
        as MVDR, MPDR, WPD, and others.

        Args:
            data (torch.complex64/ComplexTensor):
                Input tensor of shape (B, F, C, T), where B is the batch size,
                F is the number of frequency bins, C is the number of channels,
                and T is the time dimension.
            ilens (torch.Tensor):
                A tensor of shape (B,) representing the lengths of the input
                sequences.
            psd_n (torch.complex64/ComplexTensor):
                Noise covariance matrix for MVDR (shape: (B, F, C, C)),
                observation covariance matrix for MPDR/wMPDR, or stacked
                observation covariance for WPD (shape: (B, F, (btaps+1)*C,
                (btaps+1)*C)).
            psd_speech (torch.complex64/ComplexTensor):
                Speech covariance matrix (shape: (B, F, C, C)).
            psd_distortion (torch.complex64/ComplexTensor, optional):
                Distortion covariance matrix (shape: (B, F, C, C)).
            rtf_mat (torch.complex64/ComplexTensor, optional):
                RTF matrix (shape: (B, F, C, num_spk)).
            spk (int, optional):
                Speaker index. Default is 0.

        Returns:
            enhanced (torch.complex64/ComplexTensor):
                Enhanced output tensor of shape (B, F, T).
            ws (torch.complex64/ComplexTensor):
                Weight vectors of shape (B, F) or (B, F, (btaps+1)*C).

        Raises:
            ValueError: If the beamformer type is not supported.

        Examples:
            >>> beamformer = DNN_Beamformer(...)
            >>> enhanced, weights = beamformer.apply_beamforming(data, ilens, psd_n,
            ... psd_speech)

        Note:
            The implementation of beamforming is contingent upon the specified
            beamformer type and may involve various computational techniques to
            optimize performance based on the statistics provided.
        """
        # u: (B, C)
        if self.ref_channel < 0:
            u, _ = self.ref(psd_speech.to(dtype=data.dtype), ilens)
            u = u.double()
        else:
            if self.beamformer_type.endswith("_souden"):
                # (optional) Create onehot vector for fixed reference microphone
                u = torch.zeros(
                    *(data.size()[:-3] + (data.size(-2),)),
                    device=data.device,
                    dtype=torch.double
                )
                u[..., self.ref_channel].fill_(1)
            else:
                # for simplifying computation in RTF-based beamforming
                u = self.ref_channel

        if self.beamformer_type in ("mvdr", "mpdr", "wmpdr"):
            ws = self.bf_func.get_mvdr_vector_with_rtf(
                to_double(psd_n),
                to_double(psd_speech),
                to_double(psd_distortion),
                iterations=self.rtf_iterations,
                reference_vector=u,
                diagonal_loading=self.diagonal_loading,
                diag_eps=self.diag_eps,
            )
            enhanced = self.bf_func.apply_beamforming_vector(ws, to_double(data))
        elif self.beamformer_type == "mvdr_tfs":
            assert isinstance(psd_n, (list, tuple))
            ws = [
                self.bf_func.get_mvdr_vector_with_rtf(
                    to_double(psd_n_i),
                    to_double(psd_speech),
                    to_double(psd_distortion),
                    iterations=self.rtf_iterations,
                    reference_vector=u,
                    diagonal_loading=self.diagonal_loading,
                    diag_eps=self.diag_eps,
                )
                for psd_n_i in psd_n
            ]
            enhanced = stack(
                [self.bf_func.apply_beamforming_vector(w, to_double(data)) for w in ws]
            )
            with torch.no_grad():
                index = enhanced.abs().argmin(dim=0, keepdims=True)
            enhanced = enhanced.gather(0, index).squeeze(0)
            ws = stack(ws, dim=0)
        elif self.beamformer_type in (
            "mpdr_souden",
            "mvdr_souden",
            "wmpdr_souden",
        ):
            ws = self.bf_func.get_mvdr_vector(
                to_double(psd_speech),
                to_double(psd_n),
                u,
                diagonal_loading=self.diagonal_loading,
                diag_eps=self.diag_eps,
            )
            enhanced = self.bf_func.apply_beamforming_vector(ws, to_double(data))
        elif self.beamformer_type == "mvdr_tfs_souden":
            assert isinstance(psd_n, (list, tuple))
            ws = [
                self.bf_func.get_mvdr_vector(
                    to_double(psd_speech),
                    to_double(psd_n_i),
                    u,
                    diagonal_loading=self.diagonal_loading,
                    diag_eps=self.diag_eps,
                )
                for psd_n_i in psd_n
            ]
            enhanced = stack(
                [self.bf_func.apply_beamforming_vector(w, to_double(data)) for w in ws]
            )
            with torch.no_grad():
                index = enhanced.abs().argmin(dim=0, keepdims=True)
            enhanced = enhanced.gather(0, index).squeeze(0)
            ws = stack(ws, dim=0)
        elif self.beamformer_type == "wpd":
            ws = self.bf_func.get_WPD_filter_with_rtf(
                to_double(psd_n),
                to_double(psd_speech),
                to_double(psd_distortion),
                iterations=self.rtf_iterations,
                reference_vector=u,
                diagonal_loading=self.diagonal_loading,
                diag_eps=self.diag_eps,
            )
            enhanced = self.bf_func.perform_WPD_filtering(
                ws, to_double(data), self.bdelay, self.btaps
            )
        elif self.beamformer_type == "wpd_souden":
            ws = self.bf_func.get_WPD_filter_v2(
                to_double(psd_speech),
                to_double(psd_n),
                u,
                diagonal_loading=self.diagonal_loading,
                diag_eps=self.diag_eps,
            )
            enhanced = self.bf_func.perform_WPD_filtering(
                ws, to_double(data), self.bdelay, self.btaps
            )
        elif self.beamformer_type in ("mwf", "wmwf"):
            ws = self.bf_func.get_mwf_vector(
                to_double(psd_speech),
                to_double(psd_n),
                u,
                diagonal_loading=self.diagonal_loading,
                diag_eps=self.diag_eps,
            )
            enhanced = self.bf_func.apply_beamforming_vector(ws, to_double(data))
        elif self.beamformer_type == "sdw_mwf":
            ws = self.bf_func.get_sdw_mwf_vector(
                to_double(psd_speech),
                to_double(psd_n),
                u,
                denoising_weight=self.mwf_mu,
                diagonal_loading=self.diagonal_loading,
                diag_eps=self.diag_eps,
            )
            enhanced = self.bf_func.apply_beamforming_vector(ws, to_double(data))
        elif self.beamformer_type == "r1mwf":
            ws = self.bf_func.get_rank1_mwf_vector(
                to_double(psd_speech),
                to_double(psd_n),
                u,
                denoising_weight=self.mwf_mu,
                diagonal_loading=self.diagonal_loading,
                diag_eps=self.diag_eps,
            )
            enhanced = self.bf_func.apply_beamforming_vector(ws, to_double(data))
        elif self.beamformer_type in ("lcmp", "wlcmp", "lcmv"):
            ws = self.bf_func.get_lcmv_vector_with_rtf(
                to_double(psd_n),
                to_double(rtf_mat),
                reference_vector=spk,
                diagonal_loading=self.diagonal_loading,
                diag_eps=self.diag_eps,
            )
            enhanced = self.bf_func.apply_beamforming_vector(ws, to_double(data))
        elif self.beamformer_type.startswith("gev"):
            ws = self.bf_func.get_gev_vector(
                to_double(psd_n),
                to_double(psd_speech),
                mode="power",
                diagonal_loading=self.diagonal_loading,
                diag_eps=self.diag_eps,
            )
            enhanced = self.bf_func.apply_beamforming_vector(ws, to_double(data))
            if self.beamformer_type == "gev_ban":
                gain = self.bf_func.blind_analytic_normalization(ws, to_double(psd_n))
                enhanced = enhanced * gain.unsqueeze(-1)
        else:
            raise ValueError(
                "Not supporting beamformer_type={}".format(self.beamformer_type)
            )

        return enhanced.to(dtype=data.dtype), ws.to(dtype=data.dtype)

    def predict_mask(
        self, data: Union[torch.Tensor, ComplexTensor], ilens: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.LongTensor]:
        """
        Predict masks for beamforming.

        This method takes input data and estimates the masks required for
        beamforming based on the learned model parameters. The input data
        should be a complex tensor representing the signals to be processed.

        Args:
            data (torch.complex64/ComplexTensor):
                Input data of shape (B, T, C, F) where:
                - B: Batch size
                - T: Time or sequence length
                - C: Number of channels
                - F: Number of frequency bins
            ilens (torch.Tensor):
                Tensor of shape (B,) representing the actual lengths of each
                input sequence in the batch.

        Returns:
            Tuple[Tuple[torch.Tensor, ...], torch.LongTensor]:
                A tuple containing:
                - masks (torch.Tensor):
                    Estimated masks of shape (B, T, C, F) used for
                    beamforming.
                - ilens (torch.LongTensor):
                    The input lengths tensor of shape (B,).

        Examples:
            >>> model = DNN_Beamformer(...)
            >>> data = torch.randn(8, 100, 2, 256, dtype=torch.complex64)
            >>> ilens = torch.tensor([100] * 8)
            >>> masks, lengths = model.predict_mask(data, ilens)
            >>> print(masks[0].shape)  # Should print torch.Size([100, 2, 256])

        Note:
            The input data should be preprocessed and in double precision for
            optimal performance. The output masks can then be used for
            enhancing the input signals using various beamforming techniques.

        Raises:
            ValueError: If the input data dimensions do not match the expected
            shape.
        """
        masks, _ = self.mask(to_float(data.permute(0, 3, 2, 1)), ilens)
        # (B, F, C, T) -> (B, T, C, F)
        masks = [m.transpose(-1, -3) for m in masks]
        return masks, ilens


class AttentionReference(torch.nn.Module):
    """
    Attention-based reference for estimating spatial filters.

    This module utilizes an attention mechanism to generate a reference
    vector based on the power spectral density (PSD) input. It computes
    the attention weights from the input features and outputs a
    normalized reference vector for use in beamforming applications.

    Attributes:
        mlp_psd (torch.nn.Linear): Linear layer for mapping input PSD to
            attention features.
        gvec (torch.nn.Linear): Linear layer for generating the attention
            weights.
        eps (float): Small constant to avoid division by zero in
            calculations.

    Args:
        bidim (int): Input feature dimension of the PSD.
        att_dim (int): Dimension of the attention features.
        eps (float, optional): Small constant added for numerical stability
            (default: 1e-6).

    Returns:
        Tuple[torch.Tensor, torch.LongTensor]: A tuple containing the
        normalized attention vector and the input lengths.

    Examples:
        >>> attention_ref = AttentionReference(bidim=320, att_dim=128)
        >>> psd_input = torch.randn(10, 256, 4, 4)  # Example PSD input
        >>> ilens = torch.tensor([256] * 10)  # Example input lengths
        >>> u, ilens_out = attention_ref(psd_input, ilens)
        >>> print(u.shape)  # Should output (10, 4)

    Note:
        The input `psd_in` should have a shape of (B, F, C, C), where B
        is the batch size, F is the number of frequency bins, and C is
        the number of channels.
    """

    def __init__(self, bidim, att_dim, eps=1e-6):
        super().__init__()
        self.mlp_psd = torch.nn.Linear(bidim, att_dim)
        self.gvec = torch.nn.Linear(att_dim, 1)
        self.eps = eps

    def forward(
        self,
        psd_in: Union[torch.Tensor, ComplexTensor],
        ilens: torch.LongTensor,
        scaling: float = 2.0,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        DNN_Beamformer forward function.

        This method performs the forward pass of the DNN beamformer, applying
        beamforming to the input data based on the estimated masks or
        provided oracle masks. The method takes into account various
        parameters such as input lengths and power spectral densities.

        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Frequency

        Args:
            data (torch.complex64/ComplexTensor): Input data of shape
                (B, T, C, F).
            ilens (torch.Tensor): Input lengths of shape (B,).
            powers (List[torch.Tensor] or None): Optional. Used for wMPDR or
                WPD of shape (B, F, T).
            oracle_masks (List[torch.Tensor] or None): Optional. Oracle masks
                of shape (B, F, C, T). If provided, these masks will be
                used instead of self.mask.

        Returns:
            Tuple[Union[torch.Tensor, ComplexTensor], torch.LongTensor,
            torch.Tensor]: A tuple containing:
                - enhanced (torch.complex64/ComplexTensor): Enhanced output
                  of shape (B, T, F).
                - ilens (torch.Tensor): Input lengths of shape (B,).
                - masks (torch.Tensor): Estimated masks of shape
                  (B, T, C, F).

        Examples:
            >>> beamformer = DNN_Beamformer(...)
            >>> data = torch.randn(2, 100, 3, 64, dtype=torch.complex64)
            >>> ilens = torch.tensor([100, 90])
            >>> enhanced, ilens, masks = beamformer(data, ilens)

        Note:
            The input `data` can either be a standard PyTorch tensor or a
            ComplexTensor. The method performs necessary transformations to
            the input data for further processing.

        Raises:
            ValueError: If the provided `beamformer_type` is not supported
            or if any other invalid input is detected.
        """
        B, _, C = psd_in.size()[:3]
        assert psd_in.size(2) == psd_in.size(3), psd_in.size()
        # psd_in: (B, F, C, C)
        psd = psd_in.masked_fill(
            torch.eye(C, dtype=torch.bool, device=psd_in.device).type(torch.bool), 0
        )
        # psd: (B, F, C, C) -> (B, C, F)
        psd = (psd.sum(dim=-1) / (C - 1)).transpose(-1, -2)

        # Calculate amplitude
        psd_feat = (psd.real**2 + psd.imag**2 + self.eps) ** 0.5

        # (B, C, F) -> (B, C, F2)
        mlp_psd = self.mlp_psd(psd_feat)
        # (B, C, F2) -> (B, C, 1) -> (B, C)
        e = self.gvec(torch.tanh(mlp_psd)).squeeze(-1)
        u = F.softmax(scaling * e, dim=-1)
        return u, ilens
