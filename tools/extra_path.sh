#!/usr/bin/env bash
TOOL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [ -z "${TOOL_DIR}" ]; then
    echo "ERROR: Cannot derive the directory path of espnet/tools. This might be a bug."
    return 1
fi

export PATH="${TOOL_DIR}"/sph2pipe_v2.5:"${PATH:-}"
export PATH="${TOOL_DIR}"/sctk-2.4.10/bin:"${PATH:-}"
export PATH="${TOOL_DIR}"/mwerSegmenter:"${PATH:-}"
export PATH="${TOOL_DIR}"/moses/scripts/tokenizer:"${TOOL_DIR}"/moses/scripts/generic:"${TOOL_DIR}"/tools/moses/scripts/recaser/:"${TOOL_DIR}"/moses/scripts/training/"${PATH:-}"
export PATH="${TOOL_DIR}"/nkf/nkf-2.1.4:"${PATH:-}"
export PATH="${TOOL_DIR}"/PESQ/P862_annex_A_2005_CD/source:"${PATH:-}"
export PATH="${TOOL_DIR}"/kenlm/build/bin:"${PATH:-}"
export LD_LIBRARY_PATH="${TOOL_DIR}"/lib:"${TOOL_DIR}"/lib64:"${LD_LIBRARY_PATH:-}"
