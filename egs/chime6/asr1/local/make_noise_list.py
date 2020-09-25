#!/usr/bin/env python3

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import glob
import os
import sys


if len(sys.argv) != 2:
    print("Usage: {} <noises-dir>".format(sys.argv[0]))
    raise SystemExit(1)


for line in glob.glob("{}/*.wav".format(sys.argv[1])):
    fname = os.path.basename(line.strip())

    print(
        "--noise-id {} --noise-type point-source "
        "--bg-fg-type foreground {}".format(fname, line.strip())
    )
