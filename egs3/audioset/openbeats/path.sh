#!/bin/bash

export PYTHONPATH=../../../:../../TEMPLATE/openbeats:$(pwd):${PYTHONPATH:-}

source ../../../tools/activate_python.sh
source ../../../tools/extra_path.sh
