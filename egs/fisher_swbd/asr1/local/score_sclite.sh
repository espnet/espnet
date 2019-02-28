#!/bin/bash
# Copyright Johns Hopkins University (Author: Daniel Povey) 2012.  Apache 2.0.

# begin configuration section.
cmd=run.pl
stage=0

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: local/score_sclite.sh [--cmd (run.pl|queue.pl...)] <data-dir> <decode-dir> (<dict>)"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  exit 1;
fi

data=$1
dir=$2
dict=${3:-}

model=${dir}/../final.mdl # assume model one level up from decoding dir.

hubscr=${KALDI_ROOT}/tools/sctk/bin/hubscr.pl
[ ! -f ${hubscr} ] && echo "Cannot find scoring program at $hubscr" && exit 1;
hubdir=$(dirname ${hubscr})

for f in ${data}/stm ${data}/glm; do
  [ ! -f ${f} ] && echo "$0: expecting file $f to exist" && exit 1;
done

name=$(basename ${data}) # e.g. eval2000

score_dir=${dir}/scoring
ctm=${score_dir}/hyp.ctm
stm=${score_dir}/ref.stm
mkdir -p ${score_dir}
if [ ${stage} -le 0 ]; then
    ref=${dir}/ref.wrd.trn
    hyp=${dir}/hyp.wrd.trn
    if [ -z ${dict} ]; then
        # Assuming trn files exist
        trn2stm.py --orig-stm ${data}/stm ${ref} ${stm}
        trn2ctm.py ${hyp} ${ctm}
    else
        json2sctm.py ${dir}/data.json ${dict} --orig-stm ${data}/stm --stm ${stm} --refs ${ref} --ctm ${ctm} --hyps ${hyp}
    fi
fi

if [ ${stage} -le 1 ]; then
# Remove some stuff we don't want to score, from the ctm.
    cp ${ctm} ${score_dir}/tmpf
    cat ${score_dir}/tmpf | grep -i -v -E '\[NOISE|LAUGHTER|VOCALIZED-NOISE\]' | \
      grep -i -v -E '<UNK>' > ${ctm};
#     grep -i -v -E '<UNK>|%HESITATION' > $x;  # hesitation is scored
fi

# Score the set...
if [ ${stage} -le 2 ]; then
    ${cmd} ${score_dir}/score.log ${hubscr} -p ${hubdir} -V -l english -h hub5 -g ${data}/glm -r ${stm} ${ctm} || exit 1;
fi

# For eval2000 score the subsets
case "$name" in eval2000* )
  # Score only the, swbd part...
  if [ ${stage} -le 3 ]; then
        swbd_stm=${score_dir}/ref.swbd.stm
        swbd_ctm=${score_dir}/hyp.swbd.ctm
        ${cmd} ${score_dir}/score.swbd.log \
          grep -v '^en_' ${stm} '>' ${swbd_stm} '&&' \
          grep -v '^en_' ${ctm} '>' ${swbd_ctm} '&&' \
          ${hubscr} -p ${hubdir} -V -l english -h hub5 -g ${data}/glm -r ${swbd_stm} ${swbd_ctm} || exit 1;
  fi
  # Score only the, callhome part...
  if [ ${stage} -le 3 ]; then
        callhm_stm=${score_dir}/ref.callhm.stm
        callhm_ctm=${score_dir}/hyp.callhm.ctm
        ${cmd} ${score_dir}/score.callhm.log \
          grep -v '^sw_' ${stm} '>' ${callhm_stm} '&&' \
          grep -v '^sw_' ${ctm} '>' ${callhm_ctm} '&&' \
          ${hubscr} -p ${hubdir} -V -l english -h hub5 -g ${data}/glm -r ${callhm_stm} ${callhm_ctm} || exit 1;
  fi
 ;;

rt03* )

  # Score only the swbd part...
  if [ ${stage} -le 3 ]; then
        swbd_stm=${score_dir}/ref.swbd.stm
        swbd_ctm=${score_dir}/hyp.swbd.ctm
        ${cmd} ${score_dir}/score.swbd.log \
          grep -v '^fsh_' ${stm} '>' ${swbd_stm} '&&' \
          grep -v '^fsh_' ${ctm} '>' ${swbd_ctm} '&&' \
          ${hubscr} -p ${hubdir} -V -l english -h hub5 -g ${data}/glm -r ${swbd_stm} ${swbd_ctm} || exit 1;
  fi
  # Score only the fisher part...
  if [ ${stage} -le 3 ]; then
        fsh_stm=${score_dir}/ref.fsh.stm
        fsh_ctm=${score_dir}/hyp.fsh.ctm
        ${cmd} ${score_dir}/score.fsh.log \
          grep -v '^sw_' ${stm} '>' ${fsh_stm} '&&' \
          grep -v '^sw_' ${ctm} '>' ${fsh_ctm} '&&' \
          ${hubscr} -p ${hubdir} -V -l english -h hub5 -g ${data}/glm -r ${fsh_stm} ${fsh_ctm} || exit 1;
  fi
 ;;
esac

grep 'Percent Total Error' ${score_dir}/hyp.*ctm.filt.dtl

exit 0
