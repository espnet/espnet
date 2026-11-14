#!/bin/bash

export PYTHONPATH=../../../:../../TEMPLATE/esp2_slu:$(pwd):${PYTHONPATH:-}

source ../../../tools/activate_python.sh
source ../../../tools/extra_path.sh
