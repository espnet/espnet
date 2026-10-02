#!/bin/bash

export PYTHONPATH=../../../:../../TEMPLATE/svs:$(pwd):${PYTHONPATH:-}

source ../../../tools/activate_python.sh
source ../../../tools/extra_path.sh
