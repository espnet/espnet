#!/usr/bin/env bash
set -euo pipefail


if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

# Install pyopenjtalk
if [ ! -e pyopenjtalk.done ]; then
    (
        set -euo pipefail
        # Since this installer overwrites the existing pyopenjtalk, remove the done file.
        [ -e tdmelodic_pyopenjtalk.done ] && rm tdmelodic_pyopenjtalk.done
        python3 -m pip install pyopenjtalk==0.4.1 --no-cache-dir
        # pyopenjtalk fetches its dictionary lazily on first use, over plain
        # urllib and with no retry of its own, so a single reset connection
        # fails the whole install. Retry with backoff, as install_ffmpeg.sh
        # already does for its download.
        max_attempts=3
        attempt=1
        wait=5
        until python3 -c "import pyopenjtalk; pyopenjtalk.g2p('download dict')"; do
            if [ "${attempt}" -ge "${max_attempts}" ]; then
                echo "Failed to download the pyopenjtalk dictionary after" \
                     "${max_attempts} attempts"
                exit 1
            fi
            echo "Attempt ${attempt}/${max_attempts} failed," \
                 "waiting ${wait}s before retry..."
            sleep "${wait}"
            wait=$((wait * 2))
            attempt=$((attempt + 1))
        done
    )
    touch pyopenjtalk.done
else
    echo "pyopenjtalk is already installed."
fi
