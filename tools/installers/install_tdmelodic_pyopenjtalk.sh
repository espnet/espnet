#!/usr/bin/env bash
set -euo pipefail


if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

# Install pyopenjtalk
if [ ! -e tdmelodic_pyopenjtalk.done ]; then
    (
        set -euo pipefail
        # Since this installer overwrite existing pyopenjtalk, remove done file.
        [ -e pyopenjtalk.done ] && rm pyopenjtalk.done
        # TODO(kan-bayashi): Better to fix tagged version
        #   commit id when creating PR: 766477584a423a1e62b0f81f79fb7e5e189962b5
        rm -rf tdmelodic_openjtalk && git clone https://github.com/sarulab-speech/tdmelodic_openjtalk.git
        rm -rf pyopenjtalk && git clone https://github.com/r9y9/pyopenjtalk.git -b v0.1.6
        cd pyopenjtalk
        git switch -c v0.1.6
        git submodule update --recursive --init

        # concatenate the dictionary
        cp lib/open_jtalk/src/mecab-naist-jdic/naist-jdic.csv lib/open_jtalk/src/mecab-naist-jdic/naist-jdic_org.csv
        cat ../tdmelodic_openjtalk/tdmelodic_openjtalk.csv lib/open_jtalk/src/mecab-naist-jdic/naist-jdic_org.csv \
            > lib/open_jtalk/src/mecab-naist-jdic/naist-jdic.csv

        # install and check
        pip install -e .
        python3 -c "import pyopenjtalk; pyopenjtalk.g2p('download dict')"
    )
    touch tdmelodic_pyopenjtalk.done
else
    echo "tdmelodic_pyopenjtalk is already installed"
fi
