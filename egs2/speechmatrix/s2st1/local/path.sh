if ! python3 -c 'import bitarray' > /dev/null; then
    echo "Error: it seems that bitarray is not installed." >&2
    echo "Error: please install bitarray as follows." >&2
    echo "Error: pip install bitarray==2.9.2" >&2
    return 1
fi

if ! python3 -c 'import num2words' > /dev/null; then
    echo "Error: it seems that num2words is not installed." >&2
    echo "Error: please install num2words as follows." >&2
    echo "Error: pip install num2words==0.5.13" >&2
    return 1
fi

if ! ffmpeg -hide_banner -loglevel error -h 2>&1 > /dev/null; then
    echo "Error: it seems that ffmpeg is not installed." >&2
    echo "Error: please install ffmpeg as follows." >&2
    echo "Error: conda install conda-forge::ffmpeg" >&2
    return 1
fi

