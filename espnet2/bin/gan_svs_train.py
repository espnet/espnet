#!/usr/bin/env python3
from espnet2.tasks.gan_svs import GANSVSTask


def get_parser():
    parser = GANSVSTask.get_parser()
    return parser


def main(cmd=None):
    """GAN-based SVS training

    Example:

        % python gan_svs_train.py --print_config --optim1 adadelta
        % python gan_svs_train.py --config conf/train.yaml
    """
    GANSVSTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
