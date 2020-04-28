#!/bin/bash
set -euo pipefail


if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

# Install hts_engine_API
if [ ! -e hts_engine_API.done ]; then
    rm -rf hts_engine_API
    git clone https://github.com/r9y9/hts_engine_API.git
    ( 
        set -euo pipefail
        cd hts_engine_API/src && ./waf configure --prefix=../../ && ./waf build install
    )
    touch hts_engine_API.done
else
    echo "hts_engine_API is already installed"
fi

# Install open_jtalk
if [ ! -e open_jtalk.done ]; then
    rm -rf open_jtalk
    git clone https://github.com/r9y9/open_jtalk.git
    mkdir -p open_jtalk/src/build
    (
        set -euo pipefail
        cd open_jtalk/src/build && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=../../../ .. && make install
    )
    touch open_jtalk.done
else
    echo "open_jtalk is already installed"
fi

# Install pyopenjtalk
if [ ! -e pyopenjtalk.done ]; then
    rm -rf pyopenjtalk
    git clone https://github.com/r9y9/pyopenjtalk.git
    (
        set -euo pipefail
        cd pyopenjtalk && OPEN_JTALK_INSTALL_PREFIX=$(pwd) pip install -e .
    )
    touch pyopenjtalk.done
else
    echo "pyopenjtalk is already installed"
fi
