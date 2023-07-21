MAIN_ROOT=$PWD/../../..

# check extra kenlm module installation
if [ ! -d $MAIN_ROOT/tools/kenlm/build/bin ] > /dev/null; then
    echo "Error: it seems that kenlm is not installed." >&2
    echo "Error: please install kenlm as follows." >&2
    echo "Error: cd ${MAIN_ROOT}/tools && make kenlm.done" >&2
    return 1
fi
