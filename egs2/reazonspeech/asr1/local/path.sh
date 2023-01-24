if ! python3 -c 'import datasets' > /dev/null; then
    echo "ERR: Module 'datasets' is not installed." >&2
    echo "ERR: See 'https://huggingface.co/docs/datasets/quickstart'" >&2
    return 1
fi

if ! python3 -c 'import reazonspeech' > /dev/null; then
    echo "ERR: Module 'reazonspeech' is not installed." >&2
    echo "ERR: See 'https://github.com/reazon-research/ReazonSpeech'" >&2
    return 1
fi
