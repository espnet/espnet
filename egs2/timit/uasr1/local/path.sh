if [ ! -e ${MAIN_ROOT}/tools/rvad_fast.done ]; then
    echo "Error: it seems that rVADfast is not installed." >&2
    echo "Error: please install rVADfast as follows." >&2
    echo "Error: cd ${MAIN_ROOT}/tools && make rvad_fast.done" >&2
    return 1
fi

if [ ! -e ${MAIN_ROOT}/tools/kenlm.done ]; then
    echo "Error: it seems that kenlm is not installed." >&2
    echo "Error: please install kenlm as follows." >&2
    echo "Error: cd ${MAIN_ROOT}/tools && make kenlm.done" >&2
    return 1
fi

export VAD_HOME=${MAIN_ROOT}/tools/rVADfast


