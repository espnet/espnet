#!/bin/bash
folder=$1
awk -v folder="$folder" '{print $1 " " folder "/" $2}'
