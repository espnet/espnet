#!/usr/bin/env bash
# Sourced by SpeechLM launchers. Array-valued options are repeatable; each
# argument is literal. Scalars retain last-value-wins behavior. Shell config
# files may set arrays too, and explicit command-line options override them.

for ((speechlm_argpos = 1; speechlm_argpos < $#; speechlm_argpos++)); do
    if [[ ${!speechlm_argpos} == --config ]]; then
        speechlm_next=$((speechlm_argpos + 1))
        config=${!speechlm_next}
        [[ -r ${config} ]] || { echo "$0: missing config '${config}'" >&2; exit 1; }
        . "${config}"
    fi
done

speechlm_array_options=" "
while (( $# )); do
    case $1 in
        --help|-h) echo "${help_message:-No help found.}"; exit 0 ;;
        --*=*) echo "$0: use --name value, not --name=value" >&2; exit 1 ;;
        --*)
            speechlm_option=${1#--}
            speechlm_option=${speechlm_option//-/_}
            if [[ ! ${speechlm_option} =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]] ||
                ! speechlm_declaration=$(declare -p "${speechlm_option}" 2>/dev/null); then
                echo "$0: invalid option $1" >&2
                exit 1
            fi
            (( $# >= 2 )) || { echo "$0: missing value for $1" >&2; exit 1; }
            if [[ ${speechlm_declaration} == "declare -a "* ]]; then
                if [[ ${speechlm_array_options} != *" ${speechlm_option} "* ]]; then
                    declare -a "${speechlm_option}=()"
                    speechlm_array_options+="${speechlm_option} "
                fi
                speechlm_reference=${speechlm_option}[@]
                speechlm_values=("${!speechlm_reference}")
                if [[ -n $2 ]]; then
                    printf -v "${speechlm_option}[${#speechlm_values[@]}]" '%s' "$2"
                fi
            else
                speechlm_previous=${!speechlm_option}
                if [[ ${speechlm_previous} == true || ${speechlm_previous} == false ]] &&
                    [[ $2 != true && $2 != false ]]; then
                    echo "$0: expected true or false: $1 $2" >&2
                    exit 1
                fi
                printf -v "${speechlm_option}" '%s' "$2"
            fi
            shift 2
            ;;
        *) break ;;
    esac
done
