#!/usr/bin/env bash
set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

if [ ! -e visual.done ]; then
    (
        # Install gdown
        if ! python3 -c "import gdown" &> /dev/null; then
            (
                set -euo pipefail
                python3 -m pip install gdown
            )
        else
            echo "gdown is already installed."
        fi

        # Install skvideo
        if ! python3 -c "import skvideo" &> /dev/null; then
            (
                set -euo pipefail
                python3 -m pip install sk-video
            )
        else
            echo "skvideo is already installed."
        fi

        # Install skimage
        if ! python3 -c "import skimage" &> /dev/null; then
            (
                set -euo pipefail
                python3 -m pip install scikit-image
            )
        else
            echo "skimage is already installed."
        fi

        # Install cv2
        if ! python3 -c "import cv2" &> /dev/null; then
            (
                set -euo pipefail
                python3 -m pip install opencv-python
            )
        else
            echo "cv2 is already installed."
        fi

        # Install python_speech_features
        if ! python3 -c "import python_speech_features" &> /dev/null; then
            (
                set -euo pipefail
                python3 -m pip install python_speech_features
            )
        else
            echo "python_speech_features is already installed."
        fi
    )
    touch visual.done
else
    echo "visual components are already installed."
fi
