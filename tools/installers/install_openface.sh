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


# Essential Dependencies
echo "Installing Essential dependencies..."
sudo apt-get update
sudo apt-get -y install build-essential
sudo apt-get -y install g++-8
sudo apt-get -y install zip
sudo apt-get -y install libopenblas-dev liblapack-dev
sudo apt-get -y install libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev
sudo apt-get -y install libtbb2 libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
echo "Essential dependencies installed."

# Install cmake
cmake_install=$1
if [[ "$cmake_install" == INSTALL_CMAKE ]]; then
    echo "Installing specific cmake version 3.20.0"
    echo "Downloading cmake 3.20.0"
    wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz
    tar -zxvf cmake-3.20.0.tar.gz
    sudo rm -r cmake-3.20.0.tar.gz
    cd cmake-3.20.0
    sudo ./bootstrap
    make
    sudo make install
    export PATH="`pwd`/cmake-3.20.0/bin:$PATH"
    cd ..
elif [[ "$cmake_install" == INSTALL_CMAKE_PERMANENT ]]; then
    echo "Installing specific cmake version 3.20.0 and add it to path"
    echo "Downloading cmake 3.20.0"
    wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz
    tar -zxvf cmake-3.20.0.tar.gz
    sudo rm -r cmake-3.20.0.tar.gz
    cd cmake-3.20.0
    sudo ./bootstrap
    make
    sudo make install
    echo "Add cmake to path"
    DIR_CMAKE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    cd
    sudo echo \export PATH="$DIR_CMAKE/bin:\$PATH" >> .bashrc
    cd $DIR_CMAKE
    cd ..
else
    echo "Installing latest cmake version"
    sudo apt-get -y install cmake
fi

# Get OpenBLAS
sudo apt-get install libopenblas-dev

# Download and Compile OpenCV 4.1.0
echo "Downloading OpenCV..."
wget https://github.com/opencv/opencv/archive/4.1.0.zip
unzip 4.1.0.zip
cd opencv-4.1.0
mkdir -p build
cd build
echo "Installing OpenCV..."
sudo cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_TIFF=ON -D WITH_TBB=ON ..
sudo make -j$(nproc)
sudo make install
cd ../..
rm 4.1.0.zip
sudo rm -r opencv-4.1.0
echo "OpenCV installed."

# Download and Compile dlib
echo "Downloading dlib"
wget http://dlib.net/files/dlib-19.13.tar.bz2
tar xf dlib-19.13.tar.bz2
cd dlib-19.13
mkdir -p build
cd build
echo "Installing dlib"
cmake ..
cmake --build . --config Release
sudo make install
sudo ldconfig
cd ../..   
rm -r dlib-19.13.tar.bz2
echo "dlib installed"

# OpenFace installation
echo "Downloading OpenFace"
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace
echo "Installing OpenFace..."
mkdir -p build
cd build
cmake -D CMAKE_CXX_COMPILER=g++-8 -D CMAKE_C_COMPILER=gcc-8 -D CMAKE_BUILD_TYPE=RELEASE ..
make
cd ../..
echo "OpenFace successfully installed."


