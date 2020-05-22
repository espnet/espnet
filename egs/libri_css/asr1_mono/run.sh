#!/usr/bin/env bash
#
# LibriCSS monoaural baseline recipe.
#
# Copyright  2020  Johns Hopkins University (Author: Desh Raj)
# Copyright  2020  University of Stuttgart (Author: Pavel Denisov)
# Apache 2.0

# Begin configuration section.
stage=0

# Different stages
decode_stage=0

use_oracle_segments=false

# End configuration section
. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh

test_sets="dev eval"

set -e # exit on error

# please change the path accordingly
libricss_corpus=/resources/asr-data/LibriCSS

##########################################################################
# We first prepare the LibriCSS data (monoaural) in the Kaldi data
# format. We use session 0 for dev and others for eval.
##########################################################################
if [ $stage -le 0 ]; then
  local/data_prep_mono.sh $libricss_corpus
fi

#########################################################################
# ASR MODEL DOWNLOAD
# In this stage, we download Transformer models (ASR and LM)
# trained on LibriSpeech data.
#########################################################################
if [ $stage -le 1 ]; then
  local/download_asr.sh
fi

##########################################################################
# DIARIZATION MODEL DOWNLOAD
# In this stage, we download diarization models (extractor is trained on
# reverberated Voxceleb, backend is trained on Chime 6 training data).
##########################################################################
if [ $stage -le 2 ]; then
  local/download_diarizer.sh
fi

##########################################################################
# DECODING: We assume that we are just given the raw recordings (approx 10
# mins each), without segments or speaker information, so we have to decode 
# the whole pipeline, i.e., SAD -> Diarization -> ASR. This is done in the 
# local/decode.sh script.
##########################################################################
if [ $stage -le 3 ]; then
  local/decode.sh --stage $decode_stage \
    --test-sets "$test_sets" \
    --use-oracle-segments $use_oracle_segments
fi

exit 0;

