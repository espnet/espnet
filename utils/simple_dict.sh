#!/usr/bin/env bash

dict_init() {
    # Parameter safety
    if [ "$#" != 1 ]; then
        echo "Bad number of arguments."
        exit 1
    fi

    if [[ "$1" == *\/* ]]; then
        echo "Bad dictionary name."
        echo "Name: $1"
        exit 1
    fi

    # Initialize temporary directory in /tmp
    if [ -z ${simple_storage+x} ]; then
        simple_storage=$(mktemp -d)
    fi

    dict_name=$1
    mkdir "${simple_storage}/${dict_name}"
}

dict_put() {
    # Parameter safety
    if [ "$#" != 3 ]; then
        echo "Bad number of arguments."
        exit 1
    fi

    dict_name=$1
    key=$2
    value=$3

    echo ${value} > "${simple_storage}/${dict_name}/${key}"
}

dict_get() {
    # Parameter safety
    if [ "$#" != 2 ]; then
        echo "Bad number of arguments."
        exit 1
    fi

    dict_name=$1
    key=$2

    cat "${simple_storage}/${dict_name}/${key}"
}

dict_remove() {
    # Parameter safety
    if [ "$#" != 2 ]; then
        echo "Bad number of arguments."
        exit 1
    fi

    dict_name=$1
    key=$2

    rm "${simple_storage}/${dict_name}/${key}"
}
