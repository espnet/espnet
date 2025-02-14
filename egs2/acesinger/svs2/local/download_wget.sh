#!/bin/bash

# Get files from Google Drive

# $1 = file ID
# $2 = file name

gdown $1 -O $2
