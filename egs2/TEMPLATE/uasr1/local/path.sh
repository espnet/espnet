if [ ! -e ${MAIN_ROOT}/tools/rvad_fast.done ]; then
    echo "Error: it seems that rVADfast is not installed." >&2
    echo "Error: please install rVADfast as follows." >&2
    echo "Error: cd ${MAIN_ROOT}/tools && make rVADfast" >&2
    return 1
fi

export VAD_HOME=${MAIN_ROOT}/tools/rVADfast
