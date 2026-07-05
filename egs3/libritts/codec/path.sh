#!/bin/bash

export PYTHONPATH=../../../:../../TEMPLATE/codec:$(pwd):${PYTHONPATH:-}

source ../../../tools/activate_python.sh
source ../../../tools/extra_path.sh
