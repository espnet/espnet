# check extra module installation
if ! python3 -c "import g2pk" > /dev/null; then
    echo "Error: g2pk is not installed." >&2
    echo "Error: please install g2pk and its dependencies as follows:" >&2
    echo "Error: source ${MAIN_ROOT}/tools/activate_python.sh && pip install g2pK" >&2
    return 1
fi
