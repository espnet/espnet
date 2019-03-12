#!/bin/bash

# Begin configuration section.
nj=4
cmd=run.pl
compress=false
write_utt2num_frames=false # if true writes utt2num_frames
verbose=2
filetype=mat # mat or hdf5
keep_length=true
format=wav
# End configuration section.

echo "$0 $*"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
   echo "Usage: $0 [options] <data-dir> [<log-dir> [<pcm-dir>] ]";
   echo "e.g.: $0 data/train exp/dump_pcm/train pcm"
   echo "Note: <log-dir> defaults to <data-dir>/log, and <pcm-dir> defaults to <data-dir>/data"
   echo "Options: "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --write-utt2num-frames <true|false>     # If true, write utt2num_frames file."
   echo "  --filetype <mat|hdf5|sound.hdf5>                 # Specify the format of feats file"
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
  pcmdir=$3
else
  pcmdir=${data}/data
fi


# make $pcmdir an absolute pathname.
pcmdir=$(perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' ${pcmdir} ${PWD})

# use "name" as part of name of the archive.
name=$(basename ${data})

mkdir -p ${pcmdir}
mkdir -p ${logdir}

if [ -f ${data}/feats.scp ]; then
  mkdir -p ${data}/.backup
  echo "$0: moving ${data}/feats.scp to ${data}/.backup"
  mv ${data}/feats.scp ${data}/.backup
fi

scp=${data}/wav.scp

required="${scp}"

for f in ${required}; do
  if [ ! -f ${f} ]; then
    echo "$0: no such file ${f}"
  fi
done

utils/validate_data_dir.sh --no-text --no-feats ${data}

if ${write_utt2num_frames}; then
    opts="--write-num-frames=ark,t:${logdir}/utt2num_frames.JOB"
else
    opts=
fi

if [ "${filetype}" == hdf5 ];then
    ext=.h5
elif [ "${filetype}" == sound.hdf5 ];then
    ext=.flac.h5
    opts="--format ${format} "

elif [ "${filetype}" == sound ];then
    ext=
    opts="--format wav "
else
    ext=.ark
fi

if [ -f ${data}/segments ]; then
  echo "$0 [info]: segments file exists: using that."
  split_segments=""
  for n in $(seq ${nj}); do
    split_segments="${split_segments} ${logdir}/segments.${n}"
  done

  utils/split_scp.pl ${data}/segments ${split_segments}

  ${cmd} JOB=1:${nj} ${logdir}/dump_pcm_${name}.JOB.log \
      dump-pcm.py ${opts} --filetype ${filetype} --verbose=${verbose} --compress=${compress} \
      --keep-length ${keep_length} --segment=${logdir}/segments.JOB scp:${scp} \
      ark,scp:${pcmdir}/raw_pcm_${name}.JOB${ext},${pcmdir}/raw_pcm_${name}.JOB.scp

else

  echo "$0: [info]: no segments file exists: assuming wav.scp indexed by utterance."
  split_scps=""
  for n in $(seq ${nj}); do
    split_scps="${split_scps} ${logdir}/wav.${n}.scp"
  done

  utils/split_scp.pl ${scp} ${split_scps}

  ${cmd} JOB=1:${nj} ${logdir}/dump_pcm_${name}.JOB.log \
      dump-pcm.py ${opts} --filetype ${filetype} --verbose=${verbose} --compress=${compress} \
      --keep-length ${keep_length} scp:${logdir}/wav.JOB.scp \
      ark,scp:${pcmdir}/raw_pcm_${name}.JOB${ext},${pcmdir}/raw_pcm_${name}.JOB.scp

fi


# concatenate the .scp files together.
for n in $(seq ${nj}); do
  cat ${pcmdir}/raw_pcm_${name}.${n}.scp
done > ${data}/feats.scp

if ${write_utt2num_frames}; then
  for n in $(seq ${nj}); do
    cat ${logdir}/utt2num_frames.${n}
  done > ${data}/utt2num_frames
  rm ${logdir}/utt2num_frames.*
fi

rm -f ${logdir}/wav.*.scp ${logdir}/segments.* 2>/dev/null

# Write the filetype, this will be used for data2json.sh
echo ${filetype} > ${data}/filetype

nf=$(< $data/feats.scp wc -l)
nu=$(< $data/utt2spk wc -l)
if [ ${nf} -ne ${nu} ]; then
  echo "It seems not all of the feature files were successfully (${nf} != ${nu});"
  echo "consider using utils/fix_data_dir.sh ${data}"
fi

echo "Succeeded dumping pcm for ${name}"
