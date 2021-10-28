# check extra module installation
if ! python3 -c "import phonemizer" > /dev/null; then
    echo "Error: phonemizer is not installed." >&2
    echo "Error: please install phonemizer and its dependencies as follows:" >&2
    echo "Error: cd ${MAIN_ROOT}/tools && source activate_python.sh && ./installers/install_phonemizer.sh" >&2
    return 1
fi
