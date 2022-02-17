import os
import re
import sys
import argparse
import logging


def get_parser():
    parser = argparse.ArgumentParser(
        description="LRS-3 Data Preparation steps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train_val_path", type=str, help="Path to the Train/ Validation files"
    )
    parser.add_argument(
        "--test_path", type=str, help="Path to the Test files"
    )
    return parser



def main():
    parser = get_parser()
    args = parser.parse_args()
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info('debayan')
    # print('deb')
    logging.info(args)

if __name__ == '__main__':
    main()