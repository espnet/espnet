#!/usr/bin/env python3
from espnet2.tasks.sds import SDSTask


def get_parser():
    parser = SDSTask.get_parser()
    return parser


def main(cmd=None):
    r"""SDS Gradio app.

    Example:

        % python sds_app.py --config conf/train_asr.yaml
    """
    SDSTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
