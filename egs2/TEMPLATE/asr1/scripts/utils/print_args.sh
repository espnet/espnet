#!/usr/bin/env bash
set -euo pipefail

extra_chars=(
    ' '
    ';'
    '&'
    '|'
    '<'
    '>'
    '~'
    '`'
    '"'
    ' '
    '{'
    '}'
    '('
    ')'
)

new_args=""
for arg in "${@}"; do
    if [[ ${arg} = *"'"* ]]; then
        arg=$(echo "${arg}" | sed -e "s/'/'\\\\''/g")
    fi

    surround=false
    if [[ ${arg} = *\** ]]; then
        surround=true
    elif [[ ${arg} = *\?* ]]; then
        surround=true
    elif [[ ${arg} = *\\* ]]; then
        surround=true
    else
        for char in "${extra_chars[@]}"; do
            if [[ ${arg} = *${char}* ]]; then
                surround=true
                break
            fi
        done
    fi

    if "${surround}"; then
        new_args+="'${arg}' "
    else
        new_args+="${arg} "
    fi
done
echo ${new_args}
