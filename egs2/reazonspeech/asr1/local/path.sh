if ! python3 -c 'import datasets' > /dev/null; then
    echo "Error: it seems that datasets is not installed." >&2
    echo "Error: please install datasets as follows." >&2
    echo "Error: cd ${MAIN_ROOT}/tools/installers && ./install_datasets.sh" >&2
    return 1
fi
