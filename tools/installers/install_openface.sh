#!/bin/bash
#==============================================================================
# Title: install_openface.sh
# Description: Install everything necessary for OpenFace to compile. 
# Will install all required dependencies, only use if you do not have the dependencies
# already installed or if you don't mind specific versions of gcc,g++,cmake,opencv etc. installed
# Author: Fabian Hörst
# Reference: Thanks to Daniyal Shahrokhian <daniyal@kth.se>, Tadas Baltrusaitis <tadyla@gmail.com>
#            on which this script is based
# Github OpenFace: https://github.com/TadasBaltrusaitis/OpenFace
# Date: 2021-03-30
# Version : 1.0
# Usage: bash install.sh, please use just for ubuntu 18.04 or 20.04
#==============================================================================

# Exit script if any command fails
set -e 
set -o pipefail

# Get current directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check Ubuntu Version
if [ `lsb_release -d` != "18.04" ] || [ `lsb_release -d` != "20.04" ]; then
    echo "This script does not support your ubuntu Version. Please install manually. Further informations can be found here:"
    echo "https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation"
    exit 1
fi


# OpenFace installation
echo "Downloading OpenFace"
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace
rm -rf CMakeLists.txt
cd ../..
cp CMakeLists.txt installations/OpenFace
cd installations/OpenFace
echo "Installing OpenFace..."
mkdir -p build
cd build
cmake -D CMAKE_CXX_COMPILER=g++-8 -D CMAKE_C_COMPILER=gcc-8 -D CMAKE_BUILD_TYPE=RELEASE ..
make

./download_models.sh
cp lib/local/LandmarkDetector/model/patch_experts/cen_* build/bin/model/patch_experts/

cd ../..
echo "OpenFace successfully installed."


