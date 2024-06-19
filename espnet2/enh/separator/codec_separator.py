from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from packaging.version import parse as V

from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.enh.layers.sb_transformer_block import SBTransformerBlock
from espnet2.enh.layers.snake1d import Snake1d

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")

class CodecSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        internal_dim: int,
        num_spk: int,
        predict_noise: bool,
        mask_input: bool = True,
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        trf_num_layers: int = 16,
        trf_num_heads: int = 8,
        trf_feedforward: int = 2048,
        trf_dropout: float = 0.1,
        trf_use_positional_encoding: bool = True,
        trf_norm_before: bool = True,
        
    ):
        """Dual-Path RNN (DPRNN) Separator

        Args:
            input_dim: input feature dimension
            num_spk: number of speakers
            activation (str): Activation function.
            activation_params (dict): Parameters to provide to the activation function
            
        """
        super().__init__()

        self._num_spk = num_spk
        self.predict_noise = predict_noise
        self.num_outputs = self.num_spk + 1 if self.predict_noise else self.num_spk
        self.mask_input = mask_input


        self.block = SBTransformerBlock(
            num_layers = trf_num_layers,
            d_model = internal_dim,
            nhead = trf_num_heads,
            d_ffn = trf_feedforward,
            dropout = trf_dropout,  
            use_positional_encoding = trf_use_positional_encoding,
            norm_before = trf_norm_before,
        )

        self.input_dim = input_dim #this is dependent on the dac model
        self.internal_dim = internal_dim #this is up to us
        
        self.ch_down = nn.Conv1d(input_dim, internal_dim,1,bias=False) #change the dimensions for separator
        self.ch_up = nn.Conv1d(internal_dim, input_dim,1,bias=False) #return to the original dimension
        
        self.masker = weight_norm(nn.Conv1d(input_dim, input_dim*self.num_outputs, 1, bias=False)) #computes masks of the original input

        if activation == "Snake":
            self.activation = Snake1d(input_dim)
        else:
            act = getattr(nn, activation)
            self.activation = act(**activation_params)
        
        # gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, 1), #nn.ReLU() #Snake1d(input_dim) #nn.Tanh() #, Snake1d(channels)#
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, 1), nn.Sigmoid()
        )
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[Tuple[torch.Tensor], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor): Encoded feature [B, T, N]
            ilens (torch.Tensor): input lengths [Batch]
            additional (Dict or None): other data included in model
                NOTE: not used in this model

        Returns:
            masked (List[Union(torch.Tensor)]): [(B, T, N), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """
        B, T, N = input.shape
        x = input
        #[B,T,N]
        x = x.permute(0,2,1)
        #[B,N,T]
        x = self.ch_down(x)
        #[B,Nb,T]
        x = x.permute(0,2,1)
        #[B,T,Nb]
        x = self.block(x)
        #[B,T,Nb]
        x = x.permute(0,2,1)
        #[B,Nb,T]        
        x = self.ch_up(x)
        #[B,N,T]
        assert x.permute(0,2,1).shape == input.shape
        #[B, N, T]
        
        ##OLD VERSION
        # masks = self.masker(x) 
        # #[B,N*num_outputs,T]
        # masks = masks.view(B, N, self.num_outputs,T).permute(0,1,3,2)
        # masks = [
        #     self.activation(self.output(masks[...,i]) * self.output_gate(masks[...,i])).permute(0,2,1)
        #     for i in range(self.num_outputs)
        # ]
        
        B, N, T = x.shape
        masks = self.masker(x)
        
        #[B,N*num_outputs,T]
        masks = masks.view(B*self.num_outputs,-1,T)

        #[B*num_outputs, N, T]
        masks = self.output(masks) * self.output_gate(masks)
        masks = self.activation(masks)
        
        #[B*num_outputs, N, T]
        masks = masks.view(B, self.num_outputs, N, T)
        
        masks = [
            masks[:,i,:,:].permute(0,2,1)
            for i in range(self.num_outputs)
        ]
        
        if self.predict_noise:
            *masks, mask_noise = masks

        if self.mask_input:
            masked = [input * m for m in masks]
        else:
            masked = [m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        if self.predict_noise:
            if self.mask_input:
                others["noise1"] = input * mask_noise
            else:
                others["noise1"] = mask_noise

        return masked, ilens, others

    @property
    def num_spk(self):
        return self._num_spk
