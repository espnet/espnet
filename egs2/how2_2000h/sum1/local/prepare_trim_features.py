#! /bin/bash

# Copyright 2022 Roshan Sharma (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script prepares the trim feature directories based on a specified trim length 
"""
import os
import sys

trim_length = sys.argv[1] if len(sys.argv) > 1 else 100
