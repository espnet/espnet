#!/usr/bin/env bash
set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi
unames="$(uname -s)"
if [[ ! ${unames} =~ Linux && ! ${unames} =~ Darwin ]]; then
    echo "Warning: This script may not work with ${unames}. Exit with doing nothing"
    exit 0
fi

# Install festival
if [ ! -e festival.done ]; then
    rm -rf speech_tools
    # NOTE(kan-bayashi): It is better to use fixed tag
    git clone --depth 5 https://github.com/festvox/speech_tools.git
    (
        set -euo pipefail
        cd speech_tools && ./configure && make

    )
    rm -rf festival
    # NOTE(kan-bayashi): It is better to use fixed tag
    git clone --depth 5 https://github.com/festvox/festival.git
    (
        set -euo pipefail
        cd festival && ./configure --prefix=$PWD && make && make install

    )
    touch festival.done
else
    echo "festival is already installed"
fi

# Install espeak-ng
if [ ! -e espeak-ng.done ]; then
    rm -rf espeak-ng
    # NOTE(kan-bayashi): It is better to use fixed tag
    git clone https://github.com/espeak-ng/espeak-ng.git
    (
        set -euo pipefail
        cd espeak-ng && ./autogen.sh && ./configure --prefix=$PWD && make && make install

    )
    touch espeak-ng.done
else
    echo "espeak-ng is already installed"
fi

# Install MBROLA
if [ ! -e MBROLA.done ]; then
    rm -rf MBROLA
    # NOTE(kan-bayashi): It is better to use fixed tag
    git clone https://github.com/numediart/MBROLA.git
    (
        set -euo pipefail
        cd MBROLA && make
    )
    touch MBROLA.done
else
    echo "MBROLA is already installed"
fi

# Install phonemizer
if [ ! -e phonemizer.done ]; then
    (
        set -euo pipefail
        pip install phonemizer==3.0
    )
    touch phonemizer.done
else
    echo "phonemizer is already installed"
fi
