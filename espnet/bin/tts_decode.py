#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
import sys

from espnet.bin.bin_utils import check_and_prepare_env
from espnet.bin.bin_utils import get_recog_argparser


def main(args):
    parser = get_recog_argparser('tts')
    # general configuration
    parser.add_argument('--out', type=str, required=True,
                        help='Output filename')
    parser.add_argument('--preprocess-conf', type=str, default=None,
                        help='The configuration file for the pre-processing')
    # task related
    parser.add_argument('--json', type=str, required=True,
                        help='Filename of train label data (json)')
    # decoding related
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold value in decoding')
    args = parser.parse_args(args)

    check_and_prepare_env(args)

    # extract
    logging.info('backend = ' + args.backend)
    if args.backend == "pytorch":
        from espnet.tts.pytorch_backend.tts import decode
        decode(args)
    else:
        raise NotImplementedError("Only pytorch is supported.")


if __name__ == '__main__':
    main(sys.argv[1:])
