#!/bin/bash
# Script to be used with fairseq to extract discrete speech features

set -e
# data_dir is the directory containing all audio files
data_dir=$1
# split is the name of the output files, can be same as data_dir
split=$2

date
export PYTHONPATH=.
layer=6
ext=".wav"

logfile=${data_dir}/log
mkdir -p ${data_dir}
( echo '#' Running on `hostname`
  echo '#' Started at `date`
) >${logfile}

# Change to proper location
ckpt_path="hubert_base_ls960.pt"
km_path="km.bin"

# Dummy nshard setting for 1 gpu
nshard=1
rank=0

time1=`date +"%s"`
echo '#' Accounting: begin_time=$time1 >> ${logfile}

# Generate file list
key_file=${data_dir}/${split}_file_list
ls ${data_dir}/*${ext} > ${key_file}

# Generate tsv
PYTHONPATH=. python examples/hubert/simple_kmeans/generate_tsv_from_list.py -i ${key_file} -o ${data_dir}/${split}.tsv &>> ${logfile}

# dump hubert feature
echo 'Generating acoustic features: '
PYTHONPATH=. python examples/hubert/simple_kmeans/dump_hubert_feature.py ${data_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${data_dir} >>${logfile}
       
# Get quantized feature
echo 'Generating quantized ${k} features: '
PYTHONPATH=. python examples/hubert/simple_kmeans/dump_km_label.py ${data_dir} ${split} ${km_path} ${nshard} ${rank} ${data_dir}  >>${logfile}

sed '1d' ${data_dir}/${split}.tsv | awk '{n=split($1, lst, "/"); uttname=lst[n]; gsub(/\.wav|\.flac/, "", uttname); print(uttname)}' > ${data_dir}/${split}.keys
paste ${data_dir}/${split}.keys ${data_dir}/${split}_${rank}_${nshard}.km > ${data_dir}/${split}.txt

time2=`date +"%s"`
echo '#' Accounting: end_time=$time2 >> ${logfile}
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >> ${logfile}
