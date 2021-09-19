#!/usr/bin/env python3

# Copyright 2021 Yifan Peng
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Speech Commands Dataset: https://arxiv.org/abs/1804.03209
# Our data preparation is similar to the TensorFlow script:
# https://www.tensorflow.org/datasets/catalog/speech_commands


import argparse

parser = argparse.ArgumentParser(description="Process speech commands dataset.")
parser.add_argument(
    '--data_path', 
    type=str, 
    default='downloads/speech_commands_v0.02', 
    help='folder containing the original data'
)
parser.add_argument(
    '--test_data_path',
    type=str,
    default='downloads/speech_commands_test_set_v0.02',
    help='folder containing the test set'
)
args = parser.parse_args()

