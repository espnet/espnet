#!/bin/bash

if [ $# -lt 3 ]; then
    echo "Usage: $0 <data-base-path> <lrs3-username> <lrs3-password>"
    echo "--args <data-base-path> : The path where to download the dataset"
    echo "--args <iam-username> : The username required to download the dataset"
    echo "--args <iam-password> : The password required to download the dataset"
    echo "If you do not have a username/password, please request from: https://fki.tic.heia-fr.ch/register"
    exit 1
fi

download_dir=$1
username=$2
password=$3
iam_base_url=https://fki.tic.heia-fr.ch/DBs/iamDB/data/
iam_labels=ascii.tgz
iam_data=lines.tgz
iam_splits_base_url=https://raw.githubusercontent.com/shonenkov/IAM-Splitting/master/IAM-B/

echo "Logging in with given username and password"

# Need to login first and save cookies to a file, method from
# https://stackoverflow.com/questions/64715260/downloading-fki-iam-handwriting-database-files-using-the-new-2020-fall-interfa
if ! wget -nv --save-cookies ${download_dir}/cookies.txt \
    --keep-session-cookies \
    --no-check-certificate \
    --post-data "email=${username}&password=${password}" \
    --delete-after \
    https://fki.tic.heia-fr.ch/login; then
    echo "$0: Error logging in to https://fki.tic.heia-fr.ch/login"
    exit 1
fi

if ! grep -q "session" ${download_dir}/cookies.txt; then
    echo "$0: Failed to login, username or password may be incorrect"
    exit 1
fi

echo "Downloading IAM labels from ${iam_base_url}${iam_labels}"
if [ -f ${download_dir}/${iam_labels} ]; then
    rm ${download_dir}/${iam_labels}
fi
if ! wget --load-cookies ${download_dir}/cookies.txt --no-check-certificate -P ${download_dir} ${iam_base_url}${iam_labels}; then
    echo "$0: Error executing wget for IAM ${iam_labels}"
    exit 1
fi

echo "Downloading IAM image data from ${iam_base_url}${iam_lines}"
if [ -f ${download_dir}/${iam_data} ]; then
    rm ${download_dir}/${iam_data}
fi
if ! wget --load-cookies ${download_dir}/cookies.txt --no-check-certificate -P ${download_dir} ${iam_base_url}${iam_data}; then
    echo "$0: Error executing wget for IAM ${iam_data}"
    exit 1
fi

if [ -e ${download_dir}/lines.txt ]; then
    echo "File ${download_dir}/lines.txt already exists, removing it"
    rm ${download_dir}/lines.txt
fi

echo "Extracting IAM labels"
tar xzf ${download_dir}/${iam_labels} -C ${download_dir} lines.txt

if [ -e ${download_dir}/lines ]; then
    echo "Directory ${download_dir}/lines already exists, removing it"
    rm -rf ${download_dir}/lines
fi

echo "Extracting IAM lines data"
mkdir ${download_dir}/lines
tar xzf ${download_dir}/${iam_data} -C ${download_dir}/lines

echo "Downloading train/dev/test splits"
# Using the IAM-B splits from https://github.com/shonenkov/IAM-Splitting for this recipe
for split in train valid test; do
    if [ -e ${download_dir}/${split}.txt ]; then
        rm ${download_dir}/${split}.txt 
    fi
    if ! wget -nv ${iam_splits_base_url}${split}.txt -P ${download_dir}; then
        echo "Error downloading ${iam_splits_base_url}${split}.txt"
        exit 1
    fi
done

echo "Cleaning up archives and unneccesary files"
for file in ${iam_labels} ${iam_data} cookies.txt; do
    rm ${download_dir}/${file}
done

echo "$0: Successfully downloaded and extracted the IAM dataset"
