# check extra module installation
while IFS= read -r line; do
    if ! python -c "import ${line}" &> /dev/null; then
        echo "Error: it seems that ${line} is not installed." >&2
        echo "Error: please install ${line} as follows." >&2
        echo "Error: pip install ${line}" >&2
        exit 1
    fi
done < local/requirements.txt
