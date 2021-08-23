# check extra module installation
if ! python3 -c "import pyopenjtalk" > /dev/null; then
    echo "Error: pyopenjtalk is not installed." >&2
    echo "Error: please install pyopenjtalk and its dependencies as follows:" >&2
    echo "Error: cd ${MAIN_ROOT}/tools && make pyopenjtalk.done" >&2
    return 1
fi
