#!/bin/bash
#==============================================================================
# Title: install_espnet.sh
# Description: Install everything necessary for ESPnet to compile.
# Will install all required dependencies, only use if you do not have the dependencies
# Author: Fabian Hörst
# Github Vidaug: https://github.com/okankop/vidaug
# Date: 2021-07-19
# Version : 1.0
# Usage: bash install_vidaug.sh PATH_TO_ESPNET_MAIN FOLDER, please use just for ubuntu 18.04 or 20.04
#==============================================================================

# Get ESPNET Path, e.g. "/home/fabian/AVSR/espnet" from parameter handover
ESPNET=$1
. "${ESPNET}"/tools/activate_python.sh

# Install required packages
pip3 install numpy
pip3 install scipy
pip3 install scikit-image
pip3 install pillow

git clone https://github.com/okankop/vidaug
cd vidaug
python3 setup.py sdist && pip3 install dist/vidaug-0.1.tar.gz
