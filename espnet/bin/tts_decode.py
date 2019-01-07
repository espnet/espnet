#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
import os
import sys

from espnet.bin.bin_utils import check_cuda_visible_devices
from espnet.bin.bin_utils import get_recog_argparser
from espnet.bin.bin_utils import set_logging_level
from espnet.bin.bin_utils import set_seed


def main(args):
    parser = get_recog_argparser('tts')
    # general configuration
    parser.add_argument('--out', type=str, required=True,
                        help='Output filename')
    # task related
    parser.add_argument('--json', type=str, required=True,
                        help='Filename of train label data (json)')
    # decoding related
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold value in decoding')
    args = parser.parse_args(args)

    set_logging_level(args.verbose)

    check_cuda_visible_devices(args.ngpu)

    set_seed(args.seed)

    # display PYTHONPATH
    logging.info('python path = ' + os.environ.get('PYTHONPATH', '(None)'))

    # extract
    logging.info('backend = ' + args.backend)
    if args.backend == "pytorch":
        from espnet.tts.pytorch_backend.tts import decode
        decode(args)
    else:
        raise NotImplementedError("Only pytorch is supported.")


if __name__ == '__main__':
    main(sys.argv[1:])
