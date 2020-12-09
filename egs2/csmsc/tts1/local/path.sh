# check extra module installation
if ! command -v nkf > /dev/null; then
    echo "Error: it seems that nkf is not installed." >&2
    echo "Error: please install nkf as follows." >&2
    echo "Error: cd ${MAIN_ROOT}/tools && make nkf.done" >&2
    return 1
fi
