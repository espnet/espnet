if [ ! which flac &> /dev/null ]
then
    echo "Error: flac is not installed"
    return 1
fi
