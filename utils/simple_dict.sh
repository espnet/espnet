#!/usr/bin/env bash

_check_parameter_count() {
    if [ $1 != $2 ]; then
        echo "Bad number of arguments."
        exit 1
    fi
}

_check_parameter_space_or_slash() {
    if [[ "$1" == */* ]] || [[ "$1" == *\ * ]]; then
        echo "Bad name: $1"
        exit 1
    fi
}

dict_init() {
    _check_parameter_count 1 "$#"

    dict_name=$1

    _check_parameter_space_or_slash "${dict_name}"

    # Initialize temporary directory in /tmp
    if [ -z ${simple_storage+x} ]; then
        simple_storage=$(mktemp -d)
    fi
    mkdir "${simple_storage}/${dict_name}"
}

dict_put() {
    _check_parameter_count 3 "$#"

    dict_name=$1
    key=$2
    value=$3

    _check_parameter_space_or_slash "${key}"
    _check_parameter_space_or_slash "${value}"

    echo ${value} > "${simple_storage}/${dict_name}/${key}"
}

dict_get() {
    _check_parameter_count 2 "$#"

    dict_name=$1
    key=$2

    cat "${simple_storage}/${dict_name}/${key}"
}

dict_remove() {
    _check_parameter_count 2 "$#"

    dict_name=$1
    key=$2

    rm "${simple_storage}/${dict_name}/${key}"
}

dict_keys() {
    _check_parameter_count 1 "$#"

    dict_name=$1

    keys=$(ls "${simple_storage}/${dict_name}")
    echo $keys | sort
}

dict_values() {
    _check_parameter_count 1 "$#"

    dict_name=$1

    values=$(
        for key in $(dict_keys ${dict_name}); do
            dict_get $dict_name $key
        done | tr "\n" " "
    )
    echo ${values/%?/}
}
