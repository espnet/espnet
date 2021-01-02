#!/usr/bin/env bash
if [ -n "${BASH_VERSION:-}" ]; then
    # shellcheck disable=SC2046
    TOOL_DIR="$( cd $( dirname ${BASH_SOURCE[0]} ) >/dev/null 2>&1 && pwd )"
elif [ -n "${ZSH_VERSION:-}" ]; then
    # shellcheck disable=SC2046
    TOOL_DIR="$( cd $( dirname ${(%):-%N} ) >/dev/null 2>&1 && pwd )"
else
    # assume something else
    echo "ERROR: Must be executed by bash or zsh." >&2
    return 1
fi

if [ -z "${TOOL_DIR}" ]; then
    echo "ERROR: Cannot derive the directory path of espnet/tools. This might be a bug." >&2
    return 1
fi

echo "Warning! You haven't set Python environment yet. Go to ${TOOL_DIR} and generate 'activate_python.sh'" >&2
