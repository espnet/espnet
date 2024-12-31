if ! python3 -c "import s3prl" > /dev/null; then
    echo "Warning: s3prl is not installed." >&2
    echo "Warning: please install s3prl and its dependencies as follows:" >&2
    echo "Warning: cd ${MAIN_ROOT}/tools && ./installers/install_s3prl.sh" >&2
    return 1
fi