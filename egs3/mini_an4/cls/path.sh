#!/bin/bash

export PYTHONPATH=../../../:../../TEMPLATE/cls:$(pwd):${PYTHONPATH:-}

source ../../../tools/activate_python.sh
source ../../../tools/extra_path.sh
