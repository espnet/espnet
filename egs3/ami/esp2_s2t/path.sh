#!/bin/bash

export PYTHONPATH=../../../:../../TEMPLATE/esp2_s2t:$(pwd):${PYTHONPATH:-}

source ../../../tools/activate_python.sh
source ../../../tools/extra_path.sh
