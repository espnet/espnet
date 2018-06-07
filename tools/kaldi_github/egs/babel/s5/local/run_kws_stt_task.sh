#!/bin/bash  
# Copyright 2013  Johns Hopkins University (authors: Yenda Trmal)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

#Simple BABEL-only script to be run on generated lattices (to produce the
#files for scoring and for NIST submission

set -e
set -o pipefail
set -u

#Begin options
min_lmwt=8
max_lmwt=12
cer=0
skip_kws=false
skip_stt=false
cmd=run.pl
max_states=150000
dev2shadow=
eval2shadow=
wip=0.5 #Word insertion penalty
#End of options

echo $0 "$@"
. utils/parse_options.sh     

if [ $# -ne 3 ]; then
  echo "Usage: $0  <arpa-lm-file> <lang-dir> <dest-dir>"
  exit 1;
fi

data_dir=$1; 
lang_dir=$2;
decode_dir=$3; 

type=normal
if [ ! -z ${dev2shadow}  ] && [ ! -z ${eval2shadow} ] ; then
  type=shadow
elif [ -z ${dev2shadow}  ] && [ -z ${eval2shadow} ] ; then
  type=normal
else
  echo "Switches --dev2shadow and --eval2shadow must be used simultaneously" > /dev/stderr
  exit 1
fi


if [ ! -f $decode_dir/.score.done ]; then 
  local/lattice_to_ctm.sh --cmd "$cmd" --word-ins-penalty $wip \
    --min-lmwt ${min_lmwt} --max-lmwt ${max_lmwt} \
    $data_dir $lang_dir $decode_dir

  if [[ "$type" == shadow* ]]; then
    local/split_ctms.sh --cmd "$cmd" --cer $cer \
      --min-lmwt ${min_lmwt} --max-lmwt ${max_lmwt}\
      $data_dir $decode_dir ${dev2shadow} ${eval2shadow}
  else
    local/score_stm.sh --cmd "$cmd"  --cer $cer \
      --min-lmwt ${min_lmwt} --max-lmwt ${max_lmwt}\
      $data_dir $lang_dir $decode_dir
  fi
  touch $decode_dir/.score.done
fi

if ! $skip_kws ; then
  if [ ! -f $decode_dir/.kws.done ]; then 
    if [[ "$type" == shadow* ]]; then
      local/shadow_set_kws_search.sh --cmd "$cmd" --max-states ${max_states} \
        --min-lmwt ${min_lmwt} --max-lmwt ${max_lmwt}\
        $data_dir $lang_dir $decode_dir ${dev2shadow} ${eval2shadow}
    else
      local/kws_search.sh --cmd "$cmd" --max-states ${max_states} \
        --min-lmwt ${min_lmwt} --max-lmwt ${max_lmwt}\
        $lang_dir $data_dir $decode_dir
    fi
    touch $decode_dir/.kws.done
  fi
fi
