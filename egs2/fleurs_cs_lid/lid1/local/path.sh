# Check the extra dependencies of this recipe (sourced by ../path.sh).
for _module in datasets s3prl; do
    if ! python3 -c "import ${_module}" > /dev/null 2>&1; then
        echo "Error: ${_module} is not installed." >&2
        echo "Error: please install it as follows:" >&2
        echo "Error: cd ${MAIN_ROOT}/tools && ./installers/install_${_module}.sh" >&2
        unset _module
        return 1
    fi
done
unset _module

# local/download_cs_fleurs.sh and local/download_cs_yodas.sh
if ! git lfs version > /dev/null 2>&1; then
    echo "Error: git-lfs is required to download the corpora." >&2
    return 1
fi
