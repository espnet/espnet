#!/bin/bash

export PYTHONPATH=../../../:../../TEMPLATE/spk:$(pwd):${PYTHONPATH:-}

source ../../../tools/activate_python.sh
source ../../../tools/extra_path.sh
