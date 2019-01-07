#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import sys

from distutils.util import strtobool

from espnet.bin.bin_utils import check_cuda_visible_devices
from espnet.bin.bin_utils import get_train_argparser
from espnet.bin.bin_utils import set_logging_level
from espnet.bin.bin_utils import set_seed


def main(args):
    parser = get_train_argparser('tts')
    # general configuration
    parser.add_argument('--minibatches', '-N', type=int, default='-1',
                        help='Process only N minibatches (for debug)')
    # task related
    parser.add_argument('--train-json', type=str, required=True,
                        help='Filename of training json')
    parser.add_argument('--valid-json', type=str, required=True,
                        help='Filename of validation json')
    # network architecture
    # encoder
    parser.add_argument('--embed_dim', default=512, type=int,
                        help='Number of dimension of embedding')
    parser.add_argument('--elayers', default=1, type=int,
                        help='Number of encoder layers')
    parser.add_argument('--eunits', '-u', default=512, type=int,
                        help='Number of encoder hidden units')
    parser.add_argument('--econv_layers', default=3, type=int,
                        help='Number of encoder convolution layers')
    parser.add_argument('--econv_chans', default=512, type=int,
                        help='Number of encoder convolution channels')
    parser.add_argument('--econv_filts', default=5, type=int,
                        help='Filter size of encoder convolution')
    # attention
    parser.add_argument('--atype', default="location", type=str,
                        choices=["forward_ta", "forward", "location"],
                        help='Type of attention mechanism')
    parser.add_argument('--adim', default=512, type=int,
                        help='Number of attention transformation dimensions')
    parser.add_argument('--aconv-chans', default=32, type=int,
                        help='Number of attention convolution channels')
    parser.add_argument('--aconv-filts', default=15, type=int,
                        help='Filter size of attention convolution')
    parser.add_argument('--cumulate_att_w', default=True, type=strtobool,
                        help="Whether or not to cumulate attention weights")
    # decoder
    parser.add_argument('--dlayers', default=2, type=int,
                        help='Number of decoder layers')
    parser.add_argument('--dunits', default=1024, type=int,
                        help='Number of decoder hidden units')
    parser.add_argument('--prenet_layers', default=2, type=int,
                        help='Number of prenet layers')
    parser.add_argument('--prenet_units', default=256, type=int,
                        help='Number of prenet hidden units')
    parser.add_argument('--postnet_layers', default=5, type=int,
                        help='Number of postnet layers')
    parser.add_argument('--postnet_chans', default=512, type=int,
                        help='Number of postnet channels')
    parser.add_argument('--postnet_filts', default=5, type=int,
                        help='Filter size of postnet')
    parser.add_argument('--output_activation', default=None, type=str, nargs='?',
                        help='Output activation function')
    # cbhg
    parser.add_argument('--use_cbhg', default=False, type=strtobool,
                        help='Whether to use CBHG module')
    parser.add_argument('--cbhg_conv_bank_layers', default=8, type=int,
                        help='Number of convoluional bank layers in CBHG')
    parser.add_argument('--cbhg_conv_bank_chans', default=128, type=int,
                        help='Number of convoluional bank channles in CBHG')
    parser.add_argument('--cbhg_conv_proj_filts', default=3, type=int,
                        help='Filter size of convoluional projection layer in CBHG')
    parser.add_argument('--cbhg_conv_proj_chans', default=256, type=int,
                        help='Number of convoluional projection channels in CBHG')
    parser.add_argument('--cbhg_highway_layers', default=4, type=int,
                        help='Number of highway layers in CBHG')
    parser.add_argument('--cbhg_highway_units', default=128, type=int,
                        help='Number of highway units in CBHG')
    parser.add_argument('--cbhg_gru_units', default=256, type=int,
                        help='Number of GRU units in CBHG')
    # model (parameter) related
    parser.add_argument('--use_speaker_embedding', default=False, type=strtobool,
                        help='Whether to use speaker embedding')
    parser.add_argument('--use_batch_norm', default=True, type=strtobool,
                        help='Whether to use batch normalization')
    parser.add_argument('--use_concate', default=True, type=strtobool,
                        help='Whether to concatenate encoder embedding with decoder outputs')
    parser.add_argument('--use_residual', default=True, type=strtobool,
                        help='Whether to use residual connection in conv layer')
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='Dropout rate')
    parser.add_argument('--zoneout', default=0.1, type=float,
                        help='Zoneout rate')
    parser.add_argument('--reduction_factor', default=1, type=int,
                        help='Reduction factor')
    # loss related
    parser.add_argument('--use_masking', default=False, type=strtobool,
                        help='Whether to use masking in calculation of loss')
    parser.add_argument('--bce_pos_weight', default=20.0, type=float,
                        help='Positive sample weight in BCE calculation (only for use_masking=True)')
    # minibatch related
    parser.add_argument('--batch_sort_key', default='shuffle', type=str,
                        choices=['shuffle', 'output', 'input'], nargs='?',
                        help='Batch sorting key')
    parser.add_argument('--maxlen-in', default=100, type=int, metavar='ML',
                        help='Batch size is reduced if the input sequence length > ML')
    parser.add_argument('--maxlen-out', default=200, type=int, metavar='ML',
                        help='Batch size is reduced if the output sequence length > ML')
    parser.add_argument('--n_iter_processes', default=0, type=int,
                        help='Number of processes of iterator')
    # optimization related
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate for optimizer')
    parser.add_argument('--eps', default=1e-6, type=float,
                        help='Epsilon for optimizer')
    parser.add_argument('--weight-decay', default=1e-6, type=float,
                        help='Weight decay coefficient for optimizer')
    parser.add_argument('--num-save-attention', default=5, type=int,
                        help='Number of samples of attention to be saved')
    args = parser.parse_args(args)

    set_logging_level(args.verbose)

    check_cuda_visible_devices(args.ngpu)

    set_seed(args.seed)

    if args.backend == "pytorch":
        from espnet.tts.pytorch_backend.tts import train
        train(args)
    else:
        raise NotImplementedError("Only pytorch is supported.")


if __name__ == "__main__":
    main(sys.argv[1:])
