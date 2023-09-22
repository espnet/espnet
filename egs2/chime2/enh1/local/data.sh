#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

help_message=$(cat << EOF
Usage: $0
EOF
)


log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


if [ ! -e "${WSJ0}" ]; then
    log "Fill the value of 'WSJ0' in db.sh"
    exit 1
fi
if [ ! -e "${CHIME2_WSJ0}" ]; then
    log "Fill the value of 'CHIME2_WSJ0' in db.sh"
    log "You can download the data from https://catalog.ldc.upenn.edu/LDC2017S10"
    exit 1
fi
if [ ! -e "${CHIME2_GRID}" ]; then
    log "Fill the value of 'CHIME2_GRID' in db.sh"
    log "You can download the data from https://catalog.ldc.upenn.edu/LDC2017S07"
    exit 1
fi

train_set="train"
valid_set="devel"
test_set="test"


# Setup wav folders
reverb=${CHIME2_WSJ0}/data/chime2-wsj0/reverberated
noisy=${CHIME2_WSJ0}/data/chime2-wsj0/isolated
grid=${CHIME2_GRID}/data/chime2-grid

if [ ! -d "$reverb" ]; then
  echo "Error: Cannot find wav directory: $reverb"
  echo "Please ensure the zip files in '$CHIME2_WSJ0' have been extracted"
  exit 1;
fi
if [ ! -d "$noisy" ]; then
  echo "Error: Cannot find wav directory: $noisy"
  echo "Please ensure the zip files in '$CHIME2_WSJ0' have been extracted"
  exit 1;
fi
if [ ! -d "$grid" ]; then
  echo "Error: Cannot find wav directory: $grid"
  echo "Please ensure the zip files in '$CHIME2_GRID' have been extracted"
  exit 1;
fi

log "Data preparation for CHiME-2 WSJ0 clean data"
local/clean_wsj0_data_prep.sh "$WSJ0"

log "Data preparation for CHiME-2 WSJ0 noisy data"
local/noisy_wsj0_data_prep.sh "$noisy"

log "Data preparation for CHiME-2 WSJ0 reverberant data"
local/reverb_wsj0_data_prep.sh "$reverb"


log "Data preparation for CHiME-2 WSJ0 ASR data"
srcdir=data/local/data
for x in test_eval92_clean test_eval92_noisy test_eval92_5k_clean test_eval92_5k_noisy dev_dt_05_clean dev_dt_05_reverb dev_dt_05_noisy dev_dt_20_clean dev_dt_20_reverb dev_dt_20_noisy train_si84_clean train_si84_reverb train_si84_noisy; do
  mkdir -p data/$x
  cp $srcdir/${x}_wav.scp data/$x/wav.scp || exit 1;
  cp $srcdir/$x.txt data/$x/text || exit 1;
  cp $srcdir/$x.spk2utt data/$x/spk2utt || exit 1;
  cp $srcdir/$x.utt2spk data/$x/utt2spk || exit 1;
  utils/filter_scp.pl data/$x/spk2utt $srcdir/spk2gender > data/$x/spk2gender || exit 1;
done


log "Data preparation for CHiME-2 Grid data"
for x in train devel test; do
  for cond in clean reverberated isolated; do
    if [ "$cond" = "reverberated" ]; then
      if [ "$x" = "test" ]; then
        continue
      fi
      name="reverb"
    elif [ "$cond" = "isolated" ]; then
      name="noisy"
    else
      if [ "$x" = "test" ] || [ "$x" = "devel" ]; then
        continue
      fi
      name="$cond"
    fi
    mkdir -p data/${x}_grid_${name}
    find "${grid}/${x}/${cond}" -name '*.wav' | sort -u | perl -e '
      while(<>) {
        m:^\S+/(\w+)\.wav$: || die "Bad line $_";
        $id = $1;
        $id =~ tr/A-Z/a-z/;
        print "$id $_";
      }
    ' | sort > data/${x}_grid_${name}/wav.scp
    if [ "$x" = "test" ] || [ "$x" = "devel" ]; then
      if [ "$cond" = "isolated" ]; then
        mv data/${x}_grid_${name}/wav.scp data/${x}_grid_${name}/wav_tmp.scp
          # shellcheck disable=SC2002
        cat data/${x}_grid_${name}/wav_tmp.scp | perl -e '
          while(<STDIN>) {
            @A=split(" ", $_);
            @B=split("/", $_);
            $abs_path_len=@B;
            $condition=$B[$abs_path_len-2];
            if ($condition eq "9dB") {$key_suffix=2;}
            elsif ($condition eq "6dB") {$key_suffix=3;}
            elsif ($condition eq "3dB") {$key_suffix=4;}
            elsif ($condition eq "0dB") {$key_suffix=5;}
            elsif ($condition eq "m3dB") {$key_suffix=6;}
            elsif ($condition eq "m6dB") {$key_suffix=7;}
            else {print STDERR "error condition $condition";}
            print $A[0].$key_suffix." ".$A[1]."\n";
          }
        ' | sort -k1 > data/${x}_grid_${name}/wav.scp
        rm data/${x}_grid_${name}/wav_tmp.scp
      fi
    fi
    if [ "$x" = "train" ]; then
      awk '{n=split($2,arr,"/"); spk=arr[n-1]; if(length(spk) == 3) {spk=spk "_";} print(spk "_" $1 " " spk)}' data/${x}_grid_${name}/wav.scp > data/${x}_grid_${name}/utt2spk
      mv data/${x}_grid_${name}/wav.scp data/${x}_grid_${name}/wav_tmp.scp
      awk '{n=split($2,arr,"/"); spk=arr[n-1]; print(spk "_" $1 " " $2)}' data/${x}_grid_${name}/wav_tmp.scp > data/${x}_grid_${name}/wav.scp
      rm data/${x}_grid_${name}/wav_tmp.scp
    elif [ "$x" = "devel" ] && [ "$cond" = "reverberated" ]; then
      awk '{print $1}' data/${x}_grid_${name}/wav.scp \
        | perl -ane 'chop; m:^..:; print "$_ s$&\n";' > data/${x}_grid_${name}/utt2spk
    else
      awk '{print $1}' data/${x}_grid_${name}/wav.scp \
        | perl -ane 'chop; m:^...:; print "$_ $&\n";' > data/${x}_grid_${name}/utt2spk
    fi
    utils/utt2spk_to_spk2utt.pl data/${x}_grid_${name}/utt2spk > data/${x}_grid_${name}/spk2utt
  done
done


log "Data preparation for CHiME-2 ENH data"
mkdir -p data/${train_set}
for f in wav.scp utt2spk; do
  for x in train_si84 train_grid; do
    cat data/${x}_noisy/${f}
  done | sort -u > data/${train_set}/${f}
done
for x in train_si84 train_grid; do
  if [ "$x" = "train_grid" ]; then
    awk 'FNR==NR {a[$1] = $2; next} {print($1 " " a[$1])}' data/${x}_reverb/wav.scp data/${x}_noisy/wav.scp
  else
    awk 'FNR==NR {key=substr($1,1,length($1)-1); a[key] = $2; next} {key=substr($1,1,length($1)-1); print($1 " " a[key])}' data/${x}_reverb/wav.scp data/${x}_noisy/wav.scp
  fi
done | sort -u > data/${train_set}/spk1.scp

mkdir -p data/${valid_set}
for f in wav.scp utt2spk; do
  for x in dev_dt_05 dev_dt_20 devel_grid; do
    cat data/${x}_noisy/${f}
  done | sort -u > data/${valid_set}/${f}
done
for x in dev_dt_05 dev_dt_20 devel_grid; do
  if [ "$x" = "devel_grid" ]; then
    awk 'FNR==NR {a[$1] = $2; next} {key=substr($1,2,length($1)-1); print($1 " " a[key])}' data/${x}_reverb/wav.scp data/${x}_noisy/wav.scp
  else
    awk 'FNR==NR {key=substr($1,1,length($1)-1); a[key] = $2; next} {key=substr($1,1,length($1)-1); print($1 " " a[key])}' data/${x}_reverb/wav.scp data/${x}_noisy/wav.scp
  fi
done | sort -u > data/${valid_set}/spk1.scp

mkdir -p data/${test_set}
for f in wav.scp utt2spk; do
  for x in test_eval92 test_eval92_5k test_grid; do
    cat data/${x}_noisy/${f}
  done | sort -u > data/${test_set}/${f}
done

for x in ${train_set} ${valid_set} ${test_set}; do
  utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
  utils/fix_data_dir.sh --utt_extra_files "spk1.scp" data/${x}
done

log "Successfully finished. [elapsed=${SECONDS}s]"
