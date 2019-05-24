# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita and Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from distutils.util import strtobool

import torch

from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet
from espnet.nets.pytorch_backend.tacotron2.decoder import Prenet as DecoderPrenet
from espnet.nets.pytorch_backend.tacotron2.encoder import Encoder as EncoderPrenet
from espnet.nets.pytorch_backend.transformer.encoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.tts_interface import TTSInterface


def subsequent_mask(size, device="cpu", dtype=torch.uint8):
    """Create mask for subsequent steps (1, size, size)

    :param int size: size of mask
    :param str device: "cpu" or "cuda" or torch.Tensor.device
    :param torch.dtype dtype: result dtype
    :rtype: torch.Tensor
    >>> subsequent_mask(3)
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]]
    """
    ret = torch.ones(size, size, device=device, dtype=dtype)
    return torch.tril(ret, out=ret)


class Transformer(TTSInterface, torch.nn.Module):
    """Transformer for TTS

    Reference: Neural Speech Synthesis with Transformer Network (https://arxiv.org/pdf/1809.08895.pdf)
    """

    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group("transformer model setting")
        group.add_argument('--eprenet_conv_layers', default=3, type=int,
                           help='Number of encoder prenet convolution layers')
        group.add_argument('--eprenet_conv_chans', default=512, type=int,
                           help='Number of encoder prenet convolution channels')
        group.add_argument('--eprenet_conv_filts', default=5, type=int,
                           help='Filter size of encoder prenet convolution')
        group.add_argument('--dprenet_layers', default=2, type=int,
                           help='Number of decoder prenet layers')
        group.add_argument('--dprenet_units', default=256, type=int,
                           help='Number of decoder prenet hidden units')
        group.add_argument('--elayers', default=12, type=int,
                           help='Number of encoder layers')
        group.add_argument('--eunits', default=2048, type=int,
                           help='Number of encoder hidden units')
        group.add_argument('--adim', default=256, type=int,
                           help='Number of attention transformation dimensions')
        group.add_argument('--aheads', default=4, type=int,
                           help='Number of heads for multi head attention')
        group.add_argument('--dlayers', default=6, type=int,
                           help='Number of decoder layers')
        group.add_argument('--dunits', default=2048, type=int,
                           help='Number of decoder hidden units')
        group.add_argument('--postnet_layers', default=5, type=int,
                           help='Number of postnet layers')
        group.add_argument('--postnet_chans', default=512, type=int,
                           help='Number of postnet channels')
        group.add_argument('--postnet_filts', default=5, type=int,
                           help='Filter size of postnet')
        group.add_argument("--transformer-init", type=str, default="pytorch",
                           choices=["pytorch", "xavier_uniform", "xavier_normal",
                                    "kaiming_uniform", "kaiming_normal"],
                           help='how to initialize transformer parameters')
        group.add_argument("--transformer-input-layer", type=str, default="conv2d",
                           choices=["conv2d", "linear", "embed"],
                           help='transformer input layer type')
        group.add_argument('--transformer-attn-dropout-rate', default=None, type=float,
                           help='dropout in transformer attention. use --dropout-rate if None is set')
        group.add_argument('--transformer-lr', default=10.0, type=float,
                           help='Initial value of learning rate')
        group.add_argument('--transformer-warmup-steps', default=25000, type=int,
                           help='optimizer warmup steps')
        group.add_argument('--transformer-length-normalized-loss', default=True, type=strtobool,
                           help='normalize loss by length')
        return parser

    @property
    def attention_plot_class(self):
        return PlotAttentionReport

    def __init__(self, idim, odim, args):
        # initialize base classes
        TTSInterface.__init__(self)
        torch.nn.Module.__init__(self)

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.embed_dim = args.embed_dim
        self.eprenet_conv_layers = args.eprenet_conv_layers
        self.eprenet_conv_filts = args.eprenet_conv_filts
        self.eprenet_conv_chans = args.eprenet_conv_chans
        self.eprenet_proj_units = args.eprenet_proj_units
        self.dprenet_layers = args.dprenet_layers
        self.dprenet_units = args.dprenet_units
        self.postnet_layers = args.postnet_layers
        self.postnet_chans = args.postnet_chans
        self.postnet_filts = args.postnet_filts
        self.adim = args.adim
        self.aheads = args.aheads
        self.elayers = args.elayers
        self.eunits = args.eunits
        self.dlayers = args.dlayers
        self.dunits = args.dunits
        self.use_batch_norm = args.use_batch_norm
        self.dropout_rate = args.dropout_rate
        if self.eprenet_dropout_rate is None:
            self.eprenet_dropout_rate = args.dropout_rate
        if self.dprenet_dropout_rate is None:
            self.dprenet_dropout_rate = args.dropout_rate
        else:
            self.dprenet_dropout_rate = args.dprenet_dropout_rate
        if args.postnet_dropout_rate is None:
            self.postnet_dropout_rate = args.dropout_rate
        else:
            self.postnet_dropout_rate = args.postnet_dropout_rate
        if args.transformer_attn_dropout_rate is None:
            self.transformer_attn_dropout_rate = args.dropout_rate
        else:
            self.transformer_attn_dropout_rate = args.transformer_attn_dropout_rate

        # define transformer encoder
        encoder_prenet = torch.nn.Sequential(
            EncoderPrenet(
                idim=self.idim,
                embed_dim=self.embed_dim,
                elayers=0,
                econv_layers=self.eprenet_conv_layers,
                econv_chans=self.eprenet_conv_chans,
                econv_filts=self.econder_prenet_conv_filts,
                proj_units=self.econder_prenet_proj_units,
                use_batch_norm=self.use_batch_norm,
                dropout=self.eprenet_dropout_rate
            ),
            torch.nn.Linear(self.eprenet_conv_chans, self.adim)
        )
        self.encoder = Encoder(
            idim=self.idim,
            attention_dim=self.adim,
            attention_heads=self.aheads,
            linear_units=self.eunits,
            num_blocks=self.elayers,
            input_layer=encoder_prenet,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.transformer_attn_dropout_rate
        )

        # define transformer decoder
        decoder_prenet = torch.nn.Sequential(
            DecoderPrenet(
                idim=self.odim,
                n_layers=self.dprenet_layers,
                n_units=self.dprenet_units,
                dropout=self.dprenet_dropout_rate
            ),
            torch.nn.Linear(self.dprenet_units, self.adim)
        )
        self.decoder = Decoder(
            odim=-1,
            attention_dim=self.adim,
            attention_heads=self.aheads,
            linear_units=self.decoder_units,
            num_blocks=self.decoder_blocks,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.transformer_attn_dropout_rate,
            input_layer=decoder_prenet,
            use_output_layer=False
        )

        # define final projection
        self.feat_out = torch.nn.Linear(self.adim, self.odim, bias=False)
        self.prob_out = torch.nn.Linear(self.adim, 1)

        # define postnet
        self.postnet = Postnet(
            idim=self.idim,
            odim=self.odim,
            n_layers=self.postnet_layers,
            n_chans=self.postnet_chans,
            n_filts=self.postnet_filts,
            use_batch_norm=self.use_batch_norm,
            dropout=self.postnet_dropout_rate
        )

        # initialize parameters
        self.reset_parameters(args)

    def reset_parameters(self, args):
        if args.transformer_init == "pytorch":
            return
        # weight init
        for p in self.parameters():
            if p.dim() > 1:
                if args.transformer_init == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(p.data)
                elif args.transformer_init == "xavier_normal":
                    torch.nn.init.xavier_normal_(p.data)
                elif args.transformer_init == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
                elif args.transformer_init == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
                else:
                    raise ValueError("Unknown initialization: " + args.transformer_init)
        # bias init
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
        # reset some modules with default init
        for m in self.modules():
            if isinstance(m, (torch.nn.Embedding, LayerNorm)):
                m.reset_parameters()
