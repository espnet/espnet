


key_file="/shared/workspaces/anakuzne/tmp/wav_test.scp"
_nj=10
split_scps=""
_logdir="/shared/workspaces/anakuzne/tmp/log"

for n in $(seq ${_nj}); do
    split_scps+=" ${_logdir}/train.${n}.scp"
done

                                                                                                                                                                                                              
/shared/50k_train/mls_english_opus/utils/split_scp.pl "${key_file}" ${split_scps}

#queue.pl JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
#                ${python} -m espnet2.bin.lm_train \
#                    --train_shape_file "${_logdir}/train.JOB.scp" \
#                    --valid_shape_file "${_logdir}/dev.JOB.scp" \
#                    --output_dir "${_logdir}/stats.JOB" 