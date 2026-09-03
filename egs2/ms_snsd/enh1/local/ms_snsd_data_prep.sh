#!/usr/bin/env bash

. ./path.sh

if [ $# -ne 1 ]; then
  echo "Usage: $0 <ms_snsd_wav>"
  echo "Expected:"
  echo "  <ms_snsd_wav>/train/{noisy,clean,noise}"
  echo "  <ms_snsd_wav>/test/{noisy,clean,noise}"
  exit 1
fi

ms_snsd_wav=$1
data=./data
tmpdir=${data}/temp_ms_snsd
rm -rf "${tmpdir}" 2>/dev/null || true
mkdir -p "${tmpdir}"

train_noisy=${ms_snsd_wav}/train/noisy
train_clean=${ms_snsd_wav}/train/clean
train_noise=${ms_snsd_wav}/train/noise

test_noisy=${ms_snsd_wav}/test/noisy
test_clean=${ms_snsd_wav}/test/clean
test_noise=${ms_snsd_wav}/test/noise

for d in "${train_noisy}" "${train_clean}" "${train_noise}" "${test_noisy}" "${test_clean}" "${test_noise}"; do
  [ -d "${d}" ] || { echo "Error: not a directory: ${d}"; exit 1; }
done

rm -rf ${data}/tr_ms_snsd ${data}/cv_ms_snsd ${data}/tt_ms_snsd 2>/dev/null || true

# ---- helper: build a kaldi data dir from a noisy wav file list ----
make_set () {
  local setname=$1
  local noisy_list=$2
  local noisy_dir=$3
  local clean_dir=$4
  local noise_dir=$5

  mkdir -p ${data}/${setname}

  # wav.scp: uttid -> fullpath(noisy)
  # uttid = basename without .wav
  awk -F/ '{
    f=$NF; sub(/\.wav$/,"",f);
    print f, $0
  }' "${noisy_list}" | sort -k1,1 > ${data}/${setname}/wav.scp

  # spk1.scp + noise1.scp created by parsing noisy filename
  # noisy pattern: noisyN_SNRdb_S_clnspN.wav
  awk -v cdir="${clean_dir}" -v ndir="${noise_dir}" '{
    utt=$1
    noisy_path=$2

    # extract clnsp id
    if (match(utt, /clnsp[0-9]+/)) {
      cid=substr(utt, RSTART, RLENGTH)
    } else {
      print "Error: cannot parse clean id from", utt > "/dev/stderr"; exit 1
    }

    # noise stem = part before "_clnsp"
    noise_stem=utt
    sub(/_clnsp[0-9]+$/, "", noise_stem)

    print utt, cdir "/" cid ".wav"   >> "'"${data}/${setname}"'/spk1.scp"
    print utt, ndir "/" noise_stem ".wav" >> "'"${data}/${setname}"'/noise1.scp"
  }' ${data}/${setname}/wav.scp

  # sanity check
  awk '{if (system("[ -f \""$2"\" ]")!=0) {print "Missing clean:", $2; exit 1}}' ${data}/${setname}/spk1.scp
  awk '{if (system("[ -f \""$2"\" ]")!=0) {print "Missing noise:", $2; exit 1}}' ${data}/${setname}/noise1.scp

  # utt2spk/spk2utt/text
  awk '{print $1, "spk1"}' ${data}/${setname}/wav.scp > ${data}/${setname}/utt2spk
  utt2spk_to_spk2utt.pl ${data}/${setname}/utt2spk > ${data}/${setname}/spk2utt
  awk '{print $1, "dummy"}' ${data}/${setname}/wav.scp > ${data}/${setname}/text
}

# ---- train/valid split (80/20) over TRAIN noisy ----
find "${train_noisy}" -name "*.wav" | sort > ${tmpdir}/train_all.flist
num=$(wc -l < ${tmpdir}/train_all.flist)
[ ${num} -gt 1 ] || { echo "Error: too few train noisy files: ${num}"; exit 1; }

train_num=$(( num * 8 / 10 ))
head -n ${train_num} ${tmpdir}/train_all.flist > ${tmpdir}/tr.flist
tail -n $((num-train_num)) ${tmpdir}/train_all.flist > ${tmpdir}/cv.flist

echo "Split train/valid: ${train_num} / $((num-train_num))"

make_set tr_ms_snsd ${tmpdir}/tr.flist "${train_noisy}" "${train_clean}" "${train_noise}"
make_set cv_ms_snsd ${tmpdir}/cv.flist "${train_noisy}" "${train_clean}" "${train_noise}"

# ---- test set from TEST noisy ----
find "${test_noisy}" -name "*.wav" | sort > ${tmpdir}/tt.flist
make_set tt_ms_snsd ${tmpdir}/tt.flist "${test_noisy}" "${test_clean}" "${test_noise}"

echo "Done. Created data/{tr,cv,tt}_ms_snsd"
