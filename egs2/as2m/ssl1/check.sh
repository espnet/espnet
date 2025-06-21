set -ex
skip_data_prep=true
if ! "${skip_data_prep}"; then 
    echo "not skipped"
else
    echo "skipped"
fi