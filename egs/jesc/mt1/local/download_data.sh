#!/usr/bin/env bash

mkdir -p db

cd db  ### Note: the rest of this script is executed from the directory 'db'.

# TED-LIUM database:
if [ ! -e split ]; then
    echo "$0: downloading JSEC data (it won't re-download if it was already downloaded.)"
    # the following command won't re-get it if it's already there
    # because of the --continue switch.
    wget --continue --no-check-certificate https://nlp.stanford.edu/projects/jesc/data/split.tar.gz || exit 1
    tar xf "split.tar.gz"
else
    echo "$0: not downloading or un-tarring split.tar.gz because it already exists."
fi

exit 0
