from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.dnn_iva import DNN_IVA
from espnet2.enh.layers.dnn_wpe import DNN_WPE
from espnet2.enh.separator.abs_separator import AbsSeparator


class NeuralIVA(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        num_spk: int = 1,
        loss_type: str = None,
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
        # IVA options
        use_iva: bool = True,
        bnet_type: str = "blstmp",
        blayers: int = 3,
        bunits: int = 300,
        bprojs: int = 320,
        badim: int = 320,
        ref_channel: int = -1,
        bnonlinear: str = "sigmoid",
        bdropout_rate: float = 0.0,
        iva_iterations: int = 20,
        iva_train_iterations: int = None,
        iva_train_channels: int = None,
        use_dmc: bool = False,
        use_wiener: bool = False,
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

        self.use_iva = use_iva
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
        if self.use_iva:
            self.dnn_iva = DNN_IVA(
                bidim=input_dim,
                btype=bnet_type,
                blayers=blayers,
                bunits=bunits,
                bprojs=bprojs,
                num_spk=num_spk,
                nonlinear=bnonlinear,
                dropout_rate=bdropout_rate,
                badim=badim,
                ref_channel=ref_channel,
                mask_flooring=mask_flooring,
                flooring_thres=flooring_thres_bf,
                iva_iterations=iva_iterations,
                iva_train_iterations=iva_train_iterations,
                iva_train_channels=iva_train_channels,
                use_dmc=use_dmc,
                use_wiener=use_wiener,
            )
        else:
            self.dnn_iva = None

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

        powers = None
        # Performing both mask estimation and enhancement
        if input.dim() == 3:
            # single-channel input (B, T, F)
            if self.use_wpe:
                enhanced, ilens, mask_w, powers = self.wpe(input.unsqueeze(-2), ilens)
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
            if self.use_iva:
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
                    enhanced, ilens, _ = self.dnn_iva(enhanced, ilens)
                for spk in range(self.num_spk):
                    others["mask_spk{}".format(spk + 1)] = None

        if not isinstance(enhanced, list):
            enhanced = [enhanced]

        return enhanced, ilens, {}

    @property
    def num_spk(self):
        return self._num_spk
