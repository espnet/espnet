if ! which ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed"
    return 1
fi
