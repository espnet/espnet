#!/bin/bash

export PYTHONPATH=../../../:../../TEMPLATE/esp2_asr:$(pwd):${PYTHONPATH:-}

# This file is sourced: return (not exit) on failure, so that a broken
# environment is not silently replaced by whatever python3 is on PATH.
source ../../../tools/activate_python.sh || return 1
source ../../../tools/extra_path.sh || return 1
