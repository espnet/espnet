#!/bin/bash

if [ $# -ne 1 ]; then
  echo >&2 "Usage: aug2target.sh <text> > aug.tgt"
  exit 1
fi

text=$1

a=$LC_ALL
unset LC_ALL
cat ${text} | sed 's/  */ /g;s/[^ ]/& /g;s/  / <space> /g;s/  */ /g' |\
grep -v '^\s*$'
export LC_ALL=$a
