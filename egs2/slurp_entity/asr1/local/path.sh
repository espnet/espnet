# check extra module installation
if ! python3 -c "import jiwer; import textdistance; import progress" > /dev/null; then
    echo "Error: the evaluation requirements is not installed." >&2
    echo "Error: please install jiwer and its dependencies as follows:" >&2
    echo "Error: '. ${MAIN_ROOT}/activate_python.sh' (you can get the path by 'conda info --env') and 'pip install -r local/evaluation/requirements.txt' " >&2
    return 1
fi
