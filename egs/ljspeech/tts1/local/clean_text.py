#!/usr/bin/env python3

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import sys
from cleaners import english_cleaners

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('text', type=str, help='text to be cleaned')
    args = parser.parse_args()
    print(english_cleaners(args.text))
