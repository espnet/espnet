#!/bin/bash

export PYTHONPATH=../../../:../../TEMPLATE/f5tts:$(pwd):${PYTHONPATH:-}

source ../../../tools/activate_python.sh
source ../../../tools/extra_path.sh
