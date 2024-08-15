#!/bin/bash

# check extra module installation
if ! python3 -c "import transformers" > /dev/null; then
    echo "Error: pyopenjtalk is not installed." >&2
    echo "Error: please install pyopenjtalk and its dependencies as follows:" >&2
    echo "Error: cd ${MAIN_ROOT}/tools && make transformers.done" >&2
    return 1
fi
