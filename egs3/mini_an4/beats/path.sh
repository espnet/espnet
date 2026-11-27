#!/bin/bash

export PYTHONPATH=../../../:../../TEMPLATE/beats:$(pwd):${PYTHONPATH:-}

source ../../../tools/activate_python.sh
source ../../../tools/extra_path.sh
