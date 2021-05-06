#!/usr/bin/env bash
# Copyright   2019   David Snyder
#             2020   Desh Raj
# Copyright   2020   University of Stuttgart (Author: Pavel Denisov)

# Apache 2.0.
#
# This script takes an input directory that has a segments file
# and performs diarization on it. The output directory
# contains an RTTM file which can be used to resegment the input data.

stage=0
nj=10
cmd="run.pl"
diarizer_type="spectral"
score_overlaps_only=true

echo "$0 $@"  # Print the command line for logging
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
if [ $# != 4 ]; then
  echo "Usage: $0 <model-dir> <in-data-dir> <out-dir> <out-data-dir>"
  echo "e.g.: $0 exp/xvector_nnet_1a data/dev exp/dev_diarization data/dev_diarized_spectral"
  echo "Options: "
  echo "  --nj <nj>                                        # number of parallel jobs."
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --diarizer_type (agglomerative|bhmm|spectral)    # type of diarizer (default is \"spectral\")"
  echo "  --score_overlaps_only (true|false)               # provide separate scores for overlapping segments"
  exit 1;
fi

model_dir=$1
data_in=$2
out_dir=$3
data_out=$4

set -e -o pipefail

name=`basename ${data_in}`
ref_rttm=data/${name}/ref_rttm

nspk=$(wc -l < "data/${name}/spk2utt")
spknj=$(( ${nj} < ${nspk} ? ${nj} : ${nspk} ))

for f in ${data_in}/segments ${model_dir}/final.raw \
  ${model_dir}/extract.config; do
  [ ! -f $f ] && echo "$0: No such file ${f}" && exit 1;
done

if [[ ! ${diarizer_type} =~ (agglomerative|bhmm|spectral) ]]; then
  echo "$0: Unknown diarizer name: ${diarizer_type}"
  exit 0
fi

if [ $stage -le 1 ]; then
  echo "$0: computing features for x-vector extractor"
  steps/make_mfcc.sh --nj ${nj} --cmd "$cmd" \
    --mfcc-config conf/mfcc_hires.conf \
    data/${name} exp/make_mfcc/${name} mfcc
  utils/fix_data_dir.sh data/${name}
  rm -rf data/${name}_cmn
  local/nnet3/xvector/prepare_feats.sh --nj ${spknj} --cmd "$cmd" \
    data/${name} data/${name}_cmn exp/${name}_cmn
  cp data/${name}/segments exp/${name}_cmn/
  utils/fix_data_dir.sh data/${name}_cmn
fi

if [ $stage -le 2 ]; then
  echo "$0: extracting x-vectors for all segments"
  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$cmd" \
    --nj ${spknj} --window 1.5 --period 0.75 --apply-cmn false \
    --min-segment 0.5 ${model_dir} \
    data/${name}_cmn ${out_dir}/xvectors
fi

# Perform actual diarization
if [ $stage -le 3 ]; then
  echo "$0: performing ${diarizer_type} diarization"
  steps/segmentation/convert_utt2spk_and_segments_to_rttm.py data/${name}/utt2spk.bak \
    data/${name}/segments.bak ${ref_rttm}

  diar_nj=$(wc -l < "data/${name}/wav.scp") # This is important especially for VB-HMM

  if [ ${diarizer_type} == "spectral" ]; then
    echo "$0: performing cosine similarity scoring between all pairs of x-vectors"
    diarization/score_cossim.sh --cmd "$cmd" \
      --nj ${diar_nj} ${out_dir}/xvectors \
      ${out_dir}/xvectors/cossim_scores

    echo "$0: performing spectral clustering using cosine similarity scores"
    diarization/scluster.sh --cmd "$cmd" --nj ${diar_nj} \
      --rttm-channel 1 \
      ${out_dir}/xvectors/cossim_scores ${out_dir}
  else
    echo "$0: performing PLDA scoring between all pairs of x-vectors"
    diarization/nnet3/xvector/score_plda.sh --cmd "$cmd" \
      --target-energy 0.5 \
      --nj ${diar_nj} ${model_dir}/ ${out_dir}/xvectors \
      ${out_dir}/xvectors/plda_scores

    echo "$0: performing clustering using PLDA scores (threshold tuned on dev)"
    diarization/cluster.sh --cmd "$cmd" --nj ${diar_nj} \
      --rttm-channel 1 --threshold 0.4 \
      ${out_dir}/xvectors/plda_scores ${out_dir}

    if [ ${diarizer_type} == "bhmm" ]; then
      echo "$0: performing VB-HMM on top of first-pass AHC"
      diarization/vb_hmm_xvector.sh --nj ${diar_nj} --rttm-channel 1 \
        ${out_dir} ${out_dir}/xvectors ${model_dir}/plda
    fi
  fi

  echo "$0: wrote RTTM to output directory ${out_dir}"

  echo "$0 copying data files in output data directory"
  rm -rf ${data_out}
  mkdir ${data_out}
  cp ${data_in}/{wav.scp,utt2spk,utt2spk.bak,text.bak} ${data_out}
  utils/data/get_reco2dur.sh ${data_out}

  echo "$0 creating segments file from rttm and utt2spk, reco2file_and_channel "
  local/convert_rttm_to_utt2spk_and_segments.py --append-reco-id-to-spkr=true ${out_dir}/rttm \
    <(awk '{print $2" "$2" "$3}' ${out_dir}/rttm | sort -u) \
    ${data_out}/utt2spk ${data_out}/segments

  utils/utt2spk_to_spk2utt.pl ${data_out}/utt2spk > ${data_out}/spk2utt
  utils/fix_data_dir.sh ${data_out} || exit 1;

  awk '{print $1" xyz"}' ${data_out}/utt2spk > ${data_out}/text
fi

hyp_rttm=${out_dir}/rttm

# For scoring the diarization system, we use the same tool that was
# used in the DIHARD II challenge. This is available at:
# https://github.com/nryant/dscore
if [ $stage -le 4 ]; then
  echo "Diarization results for "${name}
  if ! [ -d dscore ]; then
    git clone https://github.com/desh2608/dscore.git -b libricss --single-branch || exit 1;
    cd dscore
    pip install -r requirements.txt
    cd ..
  fi

  # Create per condition ref and hyp RTTM files for scoring per condition
  mkdir -p tmp
  conditions="0L 0S OV10 OV20 OV30 OV40"
  cp $ref_rttm tmp/ref.all
  cp $hyp_rttm tmp/hyp.all
  for rttm in ref hyp; do
    for cond in $conditions; do
      cat tmp/$rttm.all | grep $cond > tmp/$rttm.$cond
    done
  done

  echo "Scoring all regions..."
  for cond in $conditions 'all'; do
    echo -n "Condition: $cond: "
    ref_rttm_path=$(readlink -f tmp/ref.$cond)
    hyp_rttm_path=$(readlink -f tmp/hyp.$cond)
    cd dscore && python score.py -r $ref_rttm_path -s $hyp_rttm_path --global_only && cd .. || exit 1;
  done

  # We also score overlapping regions only
  if [ $score_overlaps_only == "true" ]; then
    echo "Scoring overlapping regions..."
    for cond in $conditions 'all'; do
      echo -n "Condition: $cond: "
      ref_rttm_path=$(readlink -f tmp/ref.$cond)
      hyp_rttm_path=$(readlink -f tmp/hyp.$cond)
      cd dscore && python score.py -r $ref_rttm_path -s $hyp_rttm_path --overlap_only --global_only && cd .. || exit 1;
    done
  fi

  rm -r tmp
fi
