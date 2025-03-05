#!/usr/bin/env bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Copied from utils/make_fbank.sh and modified for SSL1

# Begin configuration section.
nj=4
fs=none
fmax=
fmin=
n_mels=80
n_fft=1024
n_shift=512
win_length=
window=hann
write_utt2num_frames=true
fbank_stats_file=
cmd=run.pl
compress=true
normalize=16  # The bit-depth of the input wav files
filetype=mat # mat or hdf5
# End configuration section.

use_kaldi=false # if true, use Kaldi to compute filterbank feats
python=python3       # Specify python to execute espnet commands.

help_message=$(cat <<EOF
Usage: $0 [options] <data-dir> [<log-dir> [<fbank-dir>] ]
e.g.: $0 data/train exp/make_fbank/train mfcc
Note: <log-dir> defaults to <data-dir>/log, and <fbank-dir> defaults to <data-dir>/data
Options:
  --nj <nj>                                        # number of parallel jobs
  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs.
  --filetype <mat|hdf5|sound.hdf5>                 # Specify the format of feats file
EOF
)
echo "$0 $*"  # Print the command line for logging

. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "${help_message}"
    exit 1;
fi

set -euo pipefail

data=$1
if [ $# -ge 2 ]; then
  logdir=$2
else
  logdir=${data}/log
fi
if [ $# -ge 3 ]; then
  fbankdir=$3
else
  fbankdir=${data}/data
fi

# make $fbankdir an absolute pathname.
# fbankdir=$(perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' ${fbankdir} ${PWD})

# use "name" as part of name of the archive.
name=$(basename ${data})

mkdir -p ${fbankdir} || exit 1;
mkdir -p ${logdir} || exit 1;

if [ -f ${data}/feats.scp ]; then
  mkdir -p ${data}/.backup
  echo "$0: moving $data/feats.scp to $data/.backup"
  mv ${data}/feats.scp ${data}/.backup
fi

scp=${data}/wav.scp

utils/validate_data_dir.sh --no-text --no-feats ${data} || exit 1;

split_scps=""
for n in $(seq ${nj}); do
    split_scps="${split_scps} ${logdir}/wav.${n}.scp"
done

utils/split_scp.pl ${scp} ${split_scps} || exit 1;

if ${write_utt2num_frames}; then
  write_num_frames_opt="--write-num-frames=ark,t:${logdir}/utt2num_frames.JOB"
else
  write_num_frames_opt=
fi

if [ -n ${fbank_stats_file} ]; then
  fbank_stats_file_=${fbankdir}/${fbank_stats_file%.npz}.JOB.npz
  fbank_stats_file_opt="--fbank-stats-file ${fbank_stats_file_} "
else
  fbank_stats_file_opt=
fi

if [ "${filetype}" == hdf5 ]; then
    ext=h5
else
    ext=ark
fi

if [ -f ${data}/segments ]; then
    echo "$0 [info]: segments file exists: using that."
    split_segments=""
    for n in $(seq ${nj}); do
        split_segments="${split_segments} ${logdir}/segments.${n}"
    done

    utils/split_scp.pl ${data}/segments ${split_segments}

    ${cmd} JOB=1:${nj} ${logdir}/make_fbank_${name}.JOB.log \
        ${python} -m espnet2.bin.compute_fbank_feats \
            --fs ${fs} \
            --fmax ${fmax} \
            --fmin ${fmin} \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length ${win_length} \
            --window ${window} \
            --n_mels ${n_mels} \
            ${write_num_frames_opt} \
            ${fbank_stats_file_opt} \
            --compress=${compress} \
            --use_kaldi=${use_kaldi} \
            --filetype ${filetype} \
            --normalize ${normalize} \
            --segment=${logdir}/segments.JOB scp:${scp} \
            ark,scp:${fbankdir}/raw_fbank_${name}.JOB.${ext},${fbankdir}/raw_fbank_${name}.JOB.scp

else
  echo "$0: [info]: no segments file exists: assuming pcm.scp indexed by utterance."
  split_scps=""
  for n in $(seq ${nj}); do
    split_scps="${split_scps} ${logdir}/wav.${n}.scp"
  done

  utils/split_scp.pl ${scp} ${split_scps}

  ${cmd} JOB=1:${nj} ${logdir}/make_fbank_${name}.JOB.log \
      ${python} -m espnet2.bin.compute_fbank_feats \
          --fs ${fs} \
          --fmax ${fmax} \
          --fmin ${fmin} \
          --n_fft ${n_fft} \
          --n_shift ${n_shift} \
          --win_length ${win_length} \
          --window ${window} \
          --n_mels ${n_mels} \
          ${write_num_frames_opt} \
          ${fbank_stats_file_opt} \
          --compress=${compress} \
          --use_kaldi=${use_kaldi} \
          --filetype ${filetype} \
          --normalize ${normalize} \
          scp:${logdir}/wav.JOB.scp \
          ark,scp:${fbankdir}/raw_fbank_${name}.JOB.${ext},${fbankdir}/raw_fbank_${name}.JOB.scp
fi


# concatenate the .scp files together.
for n in $(seq ${nj}); do
    cat ${fbankdir}/raw_fbank_${name}.${n}.scp || exit 1;
done > ${data}/feats.scp || exit 1

if ${write_utt2num_frames}; then
    for n in $(seq ${nj}); do
        cat ${logdir}/utt2num_frames.${n} || exit 1;
    done > ${data}/utt2num_frames || exit 1
    rm ${logdir}/utt2num_frames.* 2>/dev/null
fi

echo "Succeeded creating filterbank features!"

# Merge fbank_stats
if [ -n "${fbank_stats_file}" ]; then
    fbank_stats_files_=""
    for n in $(seq ${nj}); do
        fbank_stats_files_+="${fbankdir}/${fbank_stats_file%.npz}.${n}.npz|"
    done
    output_file="${data}/${fbank_stats_file%.npz}.txt"
    echo "Merging fbank_stats to ${output_file}"
    ${python} - <<EOF "${fbank_stats_files_}" "${output_file}"
import sys
import numpy as np
from functools import reduce
files=sys.argv[1].strip().split('|')[:-1]
output_file=sys.argv[2]
merged_data = reduce(lambda d, f: {k: d.get(k, 0) + v for k, v in np.load(f).items()}, files, {})
merged_data['fbank_mean'] = merged_data['sum'] / merged_data['count']
merged_data['fbank_std'] = np.sqrt(merged_data['sum2'] / merged_data['count'] - merged_data['fbank_mean']**2)
with open(output_file, "w") as f:
    for k, v in merged_data.items():
        f.write(f"{k}: {v:.5f}\n")
EOF
fi

rm -f ${logdir}/wav.*.scp ${logdir}/segments.* 2>/dev/null

# Write the filetype, this will be used for data2json.sh
echo ${filetype} > ${data}/filetype

nf=$(wc -l < ${data}/feats.scp)
nu=$(wc -l < ${data}/wav.scp)
if [ ${nf} -ne ${nu} ]; then
    echo "It seems not all of the feature files were successfully ($nf != $nu);"
    echo "consider using utils/fix_data_dir.sh $data"
fi

echo "Succeeded creating filterbank features for $name"
