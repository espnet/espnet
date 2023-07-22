# check extra module installation
if ! python -c 'import longformer; import nlgeval; import datasets' > /dev/null; then
    echo "Error: it seems that longformer is not installed." >&2
    echo "Error: please install longformer as follows." >&2
    echo "Error: cd ${MAIN_ROOT}/tools && make longformer.done" >&2
    return 1
fi
if ! python -c 'import nlgeval' > /dev/null; then
    echo "Error: it seems that nlgeval is not installed." >&2
    echo "Error: please install nlgeval as follows." >&2
    echo "Error: cd ${MAIN_ROOT}/tools && make longformer.done" >&2
    return 1
fi
if ! python -c 'import datasets' > /dev/null; then
    echo "Error: it seems that datasets is not installed." >&2
    echo "Error: please install datasets as follows." >&2
    echo "Error: cd ${MAIN_ROOT}/tools && make longformer.done" >&2
    return 1
fi
