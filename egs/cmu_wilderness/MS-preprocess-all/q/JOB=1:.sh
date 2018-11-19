#!/bin/bash
cd /export/b13/oadams/espnet-merge3/egs/cmu_wilderness/asr1
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
exp/train_pytorch_vggblstmp_e4_subsample1_2_2_1_1_unit512_proj512_d1_unit512_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs30_mli800_mlo150/decode_eval_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3/log/decode.JOB.log asr_recog.py --ngpu 0 --backend pytorch --recog-json dump/eval/deltafalse/splitutt/data.JOB.json --result-label exp/train_pytorch_vggblstmp_e4_subsample1_2_2_1_1_unit512_proj512_d1_unit512_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs30_mli800_mlo150/decode_eval_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3/data.JOB.json --model exp/train_pytorch_vggblstmp_e4_subsample1_2_2_1_1_unit512_proj512_d1_unit512_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs30_mli800_mlo150/results/model.model.acc.best --model-conf exp/train_pytorch_vggblstmp_e4_subsample1_2_2_1_1_unit512_proj512_d1_unit512_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs30_mli800_mlo150/results/model.json --beam-size 20 --penalty 0.0 --ctc-weight 0.3 --maxlenratio 0.0 --minlenratio 0.0 
EOF
) >JOB=1:
time1=`date +"%s"`
 ( exp/train_pytorch_vggblstmp_e4_subsample1_2_2_1_1_unit512_proj512_d1_unit512_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs30_mli800_mlo150/decode_eval_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3/log/decode.JOB.log asr_recog.py --ngpu 0 --backend pytorch --recog-json dump/eval/deltafalse/splitutt/data.JOB.json --result-label exp/train_pytorch_vggblstmp_e4_subsample1_2_2_1_1_unit512_proj512_d1_unit512_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs30_mli800_mlo150/decode_eval_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3/data.JOB.json --model exp/train_pytorch_vggblstmp_e4_subsample1_2_2_1_1_unit512_proj512_d1_unit512_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs30_mli800_mlo150/results/model.model.acc.best --model-conf exp/train_pytorch_vggblstmp_e4_subsample1_2_2_1_1_unit512_proj512_d1_unit512_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs30_mli800_mlo150/results/model.json --beam-size 20 --penalty 0.0 --ctc-weight 0.3 --maxlenratio 0.0 --minlenratio 0.0  ) 2>>JOB=1: >>JOB=1:
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>JOB=1:
echo '#' Finished at `date` with status $ret >>JOB=1:
[ $ret -eq 137 ] && exit 100;
touch ./q/sync/done.8712
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o ./q/JOB=1:  -l mem_free=4G,ram_free=4G   /export/b13/oadams/espnet-merge3/egs/cmu_wilderness/asr1/./q/JOB=1:.sh >>./q/JOB=1: 2>&1
