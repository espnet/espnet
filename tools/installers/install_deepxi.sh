#!/bin/bash
#==============================================================================
# Title: install_deepxi.sh
# Description: Install everything necessary for deepxi to compile. 
# Author: Fabian Hörst, based on DeepXi GitHub page
# Github DeepXi: https://github.com/anicolson/DeepXi
# Date: 2021-12-04
# Version : 1.0
# Usage: bash install_deepxi.sh
# Python environment: DeepXi Python environment is saved under ~/venv/DeepXi in
#	  	      your home directory
#==============================================================================

# Exit script if any command fails
set -e 
set -o pipefail

echo "Installing DeepXi"

# If Direcotry exists, pull missing files
if [ -d "DeepXi" ]; then
    cd DeepXi
    git pull https://github.com/anicolson/DeepXi.git
    cd ..
# Clone git in current directory, build virtual environment and install requirements
else
    git clone https://github.com/anicolson/DeepXi.git	
fi
python3 -m venv --system-site-packages venv/DeepXi
source venv/DeepXi/bin/activate
cd DeepXi
pip install -r requirements.txt
cd ..
echo "DeepXi installed"
