#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Nagoya University (Tomoki Hayashi)
# Modified Copyright Academia Sinica (Pin-Jui Ku)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Lip2Wav encoder related modules."""

import six

import torch

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


def encoder_init(m):
    """Initialize encoder parameters."""
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("relu"))

class Conv3dBlock(torch.nn.Module):
    """
    This Block contains 3 layers of conv3d, each has a little different 
    size of kernel and stride according to the original paper: 'balabala'   
    """

    def __init__(
        self, 
        ichan, 
        outchan,
        filts,
        use_batch_norm=True,
        use_residual=True,
        dropout_rate=0.0,
    ):
        super(Conv3dBlock, self).__init__()
        self.use_residual = use_residual
        self.convlayers = torch.nn.ModuleList()
        if use_batch_norm:
            self.convlayers += [
                torch.nn.Sequential(
                    torch.nn.Conv3d(                    
                        ichan,
                        outchan,
                        kernel_size=filts,
                        stride=(1,2,2),
                        padding=(filts-1)//2                
                    ),
                    torch.nn.BatchNorm3d(outchan),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout_rate)
                ),
                torch.nn.Sequential(
                    torch.nn.Conv3d(                    
                        outchan,
                        outchan,
                        kernel_size=filts,
                        stride=(1,1,1),
                        padding=(filts-1)//2                
                    ),
                    torch.nn.BatchNorm3d(outchan),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout_rate)
                ),
                torch.nn.Sequential(
                    torch.nn.Conv3d(                    
                        outchan,
                        outchan,
                        kernel_size=filts,
                        stride=(1,1,1),
                        padding=(filts-1)//2                
                    ),
                    torch.nn.BatchNorm3d(outchan),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout_rate)
                )                
            ]
        else:
            self.convlayers += [
                torch.nn.Sequential(
                    torch.nn.Conv3d(                    
                        ichan,
                        outchan,
                        kernel_size=filts,
                        stride=(1,2,2),
                        padding=(filts-1)//2                
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout_rate)
                ),
                torch.nn.Sequential(
                    torch.nn.Conv3d(                    
                        outchan,
                        outchan,
                        kernel_size=filts,
                        stride=(1,1,1),
                        padding=(filts-1)//2                
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout_rate)
                ),
                torch.nn.Sequential(
                    torch.nn.Conv3d(                    
                        outchan,
                        outchan,
                        kernel_size=filts,
                        stride=(1,1,1),
                        padding=(filts-1)//2                
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout_rate)
                )                
            ]

    def forward(self, xs):
        for i in six.moves.range(3):
            if self.use_residual and i != 0:
                xs = xs + self.convlayers[i](xs)
            else:
                xs = self.convlayers[i](xs)
        return xs





class Encoder(torch.nn.Module):
    """Encoder module of Spectrogram prediction network.

    This is a module of encoder of Spectrogram prediction network in Lip2Wav,
    which described in ''. This is the encoder which converts the lip image 
    sequence into the sequence of hidden states.


    """

    def __init__(
        self,
        idim,
        width,
        height,
        elayers=1,
        eunits=512,
        econv_layers=3,
        econv_chans=512,
        econv_filts=5,
        use_batch_norm=True,
        use_residual=False,
        reduce_time_length=False,
        encoder_reduction_factor=1,
        dropout_rate=0.5,
        padding_idx=0,
    ):
        """Initialize Tacotron2 encoder module.

        Args:
            idim (int) Dimension of the inputs.
            width (int) Dimension of the input width.
            height (int) Dimension of th input height.
            elayers (int, optional) The number of encoder blstm layers.
            eunits (int, optional) The number of encoder blstm units.
            econv_layers (int, optional) The number of encoder conv layers.
            econv_filts (int, optional) The number of encoder conv filter size.
            econv_chans (int, optional) The number of encoder conv filter channels.
            use_batch_norm (bool, optional) Whether to use batch normalization.
            use_residual (bool, optional) Whether to use residual connection.
            dropout_rate (float, optional) Dropout rate.

        """
        super(Encoder, self).__init__()
        # store the hyperparameters
        self.idim = idim
        self.width = width
        self.height = height*encoder_reduction_factor
        self.use_residual = use_residual
        self.reduce_time_length = reduce_time_length

        # define network layer modules
        if econv_layers > 0:
            self.convs = torch.nn.ModuleList()
            for layer in six.moves.range(econv_layers):
                ichans = (
                    1 if layer == 0 else econv_chans
                )
                self.convs += [
                    Conv3dBlock(
                        ichans,
                        econv_chans,
                        econv_filts,
                        use_batch_norm,
                        use_residual,
                        dropout_rate
                    )
                ]
        else:
            raise Exception("econv_layers can not smaller than 0!")
        
        if self.reduce_time_length:
            in_dim = econv_chans*idim // (4**econv_layers)
            out_dim = in_dim
            self.conv1d = torch.nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)

        
        if elayers > 0:
            iunits = econv_chans*idim // (4**econv_layers)
            self.blstm = torch.nn.LSTM(
                iunits, eunits // 2, elayers, batch_first=True, bidirectional=True
            )
        else:
            iunits = econv_chans*idim // (4**econv_layers)
            self.linear = torch.nn.Linear(iunits, eunits)
            self.blstm = None

        self.apply(encoder_init)


    def forward(self, xs, ilens=None):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of the padded sequence. Should Be the Video feautes (
            B, Tmax, width, height)
            ilens (LongTensor): Batch of lengths of each input batch (B,).

        Returns:
            Tensor: Batch of the sequences of encoder states(B, Tmax, eunits).
            LongTensor: Batch of lengths of each sequence (B,)

        """
        B, T, idim = xs.shape
        xs = xs.view(B, 1, T, self.width, self.height)
        for i in six.moves.range(len(self.convs)):
            xs = self.convs[i](xs)

        B, C, T, W, H = xs.shape
        xs = xs.permute(0, 1, 3, 4, 2)
        xs = xs.reshape(B, C*W*H, T)
        if self.reduce_time_length:
            xs = self.conv1d(xs)
            for i in range(len(ilens)):
                ilens[i] = ilens[i]//2
        xs = xs.transpose(1, 2)

        if self.blstm is None:
            xs = self.linear(xs)
            return xs.transpose(1, 2)
        xs = pack_padded_sequence(xs, ilens, batch_first=True)
        self.blstm.flatten_parameters()
        xs, _ = self.blstm(xs)  # (B, Tmax, C)
        xs, hlens = pad_packed_sequence(xs, batch_first=True)

        return xs, hlens

    def inference(self, x):
        """Inference.

        Args:
            x (Tensor): The sequeunce of character ids (T,)
                    or acoustic feature (T, idim * encoder_reduction_factor).

        Returns:
            Tensor: The sequences of encoder states(T, eunits).

        """
        xs = x.unsqueeze(0)
        ilens = [x.size(0)]

        return self.forward(xs, ilens)[0][0]
