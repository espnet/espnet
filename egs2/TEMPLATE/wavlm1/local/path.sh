# check extra module installation
if ! python3 -c "import fairseq" > /dev/null; then
    echo "Error: fairseq is not installed." >&2
    echo "Error: please install fairseq and its dependencies as follows:" >&2
    echo "Error: cd ${MAIN_ROOT}/tools && make fairseq.done" >&2
    return 1
fi
