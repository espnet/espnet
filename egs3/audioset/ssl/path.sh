#!/bin/bash

export PYTHONPATH=../../../:../../TEMPLATE/ssl:$(pwd):${PYTHONPATH:-}

source ../../../tools/activate_python.sh
source ../../../tools/extra_path.sh
