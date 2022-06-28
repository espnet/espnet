from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.dnn_beamformer import DNN_Beamformer
from espnet2.enh.layers.dnn_wpe import DNN_WPE
from espnet2.enh.separator.abs_separator import AbsSeparator


class NeuralBeamformer(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        num_spk: int = 1,
        loss_type: str = "mask_mse",
        # Dereverberation options
        use_wpe: bool = False,
        wnet_type: str = "blstmp",
        wlayers: int = 3,
        wunits: int = 300,
        wprojs: int = 320,
        wdropout_rate: float = 0.0,
        taps: int = 5,
        delay: int = 3,
        use_dnn_mask_for_wpe: bool = True,
        wnonlinear: str = "crelu",
        multi_source_wpe: bool = True,
        wnormalization: bool = False,
        # Beamformer options
        use_beamformer: bool = True,
        bnet_type: str = "blstmp",
        blayers: int = 3,
        bunits: int = 300,
        bprojs: int = 320,
        badim: int = 320,
        ref_channel: int = -1,
        use_noise_mask: bool = True,
        bnonlinear: str = "sigmoid",
        beamformer_type: str = "mvdr_souden",
        rtf_iterations: int = 2,
        bdropout_rate: float = 0.0,
        shared_power: bool = True,
        # For numerical stability
        diagonal_loading: bool = True,
        diag_eps_wpe: float = 1e-7,
        diag_eps_bf: float = 1e-7,
        mask_flooring: bool = False,
        flooring_thres_wpe: float = 1e-6,
        flooring_thres_bf: float = 1e-6,
        use_torch_solver: bool = True,
    ):
        super().__init__()

        self._num_spk = num_spk
        self.loss_type = loss_type
        if loss_type not in ("mask_mse", "spectrum", "spectrum_log", "magnitude"):
            raise ValueError("Unsupported loss type: %s" % loss_type)

        self.use_beamformer = use_beamformer
        self.use_wpe = use_wpe

        if self.use_wpe:
            if use_dnn_mask_for_wpe:
                # Use DNN for power estimation
                iterations = 1
            else:
                # Performing as conventional WPE, without DNN Estimator
                iterations = 2

            self.wpe = DNN_WPE(
                wtype=wnet_type,
                widim=input_dim,
                wlayers=wlayers,
                wunits=wunits,
                wprojs=wprojs,
                dropout_rate=wdropout_rate,
                taps=taps,
                delay=delay,
                use_dnn_mask=use_dnn_mask_for_wpe,
                nmask=1 if multi_source_wpe else num_spk,
                nonlinear=wnonlinear,
                iterations=iterations,
                normalization=wnormalization,
                diagonal_loading=diagonal_loading,
                diag_eps=diag_eps_wpe,
                mask_flooring=mask_flooring,
                flooring_thres=flooring_thres_wpe,
                use_torch_solver=use_torch_solver,
            )
        else:
            self.wpe = None

        self.ref_channel = ref_channel
        if self.use_beamformer:
            self.beamformer = DNN_Beamformer(
                bidim=input_dim,
                btype=bnet_type,
                blayers=blayers,
                bunits=bunits,
                bprojs=bprojs,
                num_spk=num_spk,
                use_noise_mask=use_noise_mask,
                nonlinear=bnonlinear,
                dropout_rate=bdropout_rate,
                badim=badim,
                ref_channel=ref_channel,
                beamformer_type=beamformer_type,
                rtf_iterations=rtf_iterations,
                btaps=taps,
                bdelay=delay,
                diagonal_loading=diagonal_loading,
                diag_eps=diag_eps_bf,
                mask_flooring=mask_flooring,
                flooring_thres=flooring_thres_bf,
                use_torch_solver=use_torch_solver,
            )
        else:
            self.beamformer = None

        # share speech powers between WPE and beamforming (wMPDR/WPD)
        self.shared_power = shared_power and use_wpe

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.complex64/ComplexTensor):
                mixed speech [Batch, Frames, Channel, Freq]
            ilens (torch.Tensor): input lengths [Batch]
            additional (Dict or None): other data included in model
                NOTE: not used in this model

        Returns:
            enhanced speech (single-channel): List[torch.complex64/ComplexTensor]
            output lengths
            other predcited data: OrderedDict[
                'dereverb1': ComplexTensor(Batch, Frames, Channel, Freq),
                'mask_dereverb1': torch.Tensor(Batch, Frames, Channel, Freq),
                'mask_noise1': torch.Tensor(Batch, Frames, Channel, Freq),
                'mask_spk1': torch.Tensor(Batch, Frames, Channel, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Channel, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Channel, Freq),
            ]
        """
        # Shape of input spectrum must be (B, T, F) or (B, T, C, F)
        assert input.dim() in (3, 4), input.dim()
        enhanced = input
        others = OrderedDict()

        if (
            self.training
            and self.loss_type is not None
            and self.loss_type.startswith("mask")
        ):
            # Only estimating masks during training for saving memory
            if self.use_wpe:
                if input.dim() == 3:
                    mask_w, ilens = self.wpe.predict_mask(input.unsqueeze(-2), ilens)
                    mask_w = mask_w.squeeze(-2)
                elif input.dim() == 4:
                    mask_w, ilens = self.wpe.predict_mask(input, ilens)

                if mask_w is not None:
                    if isinstance(enhanced, list):
                        # single-source WPE
                        for spk in range(self.num_spk):
                            others["mask_dereverb{}".format(spk + 1)] = mask_w[spk]
                    else:
                        # multi-source WPE
                        others["mask_dereverb1"] = mask_w

            if self.use_beamformer and input.dim() == 4:
                others_b, ilens = self.beamformer.predict_mask(input, ilens)
                for spk in range(self.num_spk):
                    others["mask_spk{}".format(spk + 1)] = others_b[spk]
                if len(others_b) > self.num_spk:
                    others["mask_noise1"] = others_b[self.num_spk]

            return None, ilens, others

        else:
            powers = None
            # Performing both mask estimation and enhancement
            if input.dim() == 3:
                # single-channel input (B, T, F)
                if self.use_wpe:
                    enhanced, ilens, mask_w, powers = self.wpe(
                        input.unsqueeze(-2), ilens
                    )
                    if isinstance(enhanced, list):
                        # single-source WPE
                        enhanced = [enh.squeeze(-2) for enh in enhanced]
                        if mask_w is not None:
                            for spk in range(self.num_spk):
                                key = "dereverb{}".format(spk + 1)
                                others[key] = enhanced[spk]
                                others["mask_" + key] = mask_w[spk].squeeze(-2)
                    else:
                        # multi-source WPE
                        enhanced = enhanced.squeeze(-2)
                        if mask_w is not None:
                            others["dereverb1"] = enhanced
                            others["mask_dereverb1"] = mask_w.squeeze(-2)
            else:
                # multi-channel input (B, T, C, F)
                # 1. WPE
                if self.use_wpe:
                    enhanced, ilens, mask_w, powers = self.wpe(input, ilens)
                    if mask_w is not None:
                        if isinstance(enhanced, list):
                            # single-source WPE
                            for spk in range(self.num_spk):
                                key = "dereverb{}".format(spk + 1)
                                others[key] = enhanced[spk]
                                others["mask_" + key] = mask_w[spk]
                        else:
                            # multi-source WPE
                            others["dereverb1"] = enhanced
                            others["mask_dereverb1"] = mask_w.squeeze(-2)

                # 2. Beamformer
                if self.use_beamformer:
                    if (
                        not self.beamformer.beamformer_type.startswith("wmpdr")
                        or not self.beamformer.beamformer_type.startswith("wpd")
                        or not self.shared_power
                        or (self.wpe.nmask == 1 and self.num_spk > 1)
                    ):
                        powers = None

                    # enhanced: (B, T, C, F) -> (B, T, F)
                    if isinstance(enhanced, list):
                        # outputs of single-source WPE
                        raise NotImplementedError(
                            "Single-source WPE is not supported with beamformer "
                            "in multi-speaker cases."
                        )
                    else:
                        # output of multi-source WPE
                        enhanced, ilens, others_b = self.beamformer(
                            enhanced, ilens, powers=powers
                        )
                    for spk in range(self.num_spk):
                        others["mask_spk{}".format(spk + 1)] = others_b[spk]
                    if len(others_b) > self.num_spk:
                        others["mask_noise1"] = others_b[self.num_spk]

        if not isinstance(enhanced, list):
            enhanced = [enhanced]

        return enhanced, ilens, others

    @property
    def num_spk(self):
        return self._num_spk
