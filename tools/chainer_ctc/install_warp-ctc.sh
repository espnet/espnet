#!/usr/bin/env bash

mkdir -p ext/warp-ctc
cd ext/warp-ctc
git clone https://github.com/baidu-research/warp-ctc.git .
mkdir build
cd build
cmake ../
make