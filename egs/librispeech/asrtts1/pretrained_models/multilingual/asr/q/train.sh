#!/bin/bash
cd /export/a08/obask/espnet_cyc/egs/jsalt18e2e/asr1
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
asr_train.py --ngpu 1 --backend pytorch --outdir exp/tr_babel10_pytorch_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_sampprob0.5_bs40_mli800_mlo150/results --debugmode 1 --dict data/lang_1char/train_units.txt --debugdir exp/tr_babel10_pytorch_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_sampprob0.5_bs40_mli800_mlo150 --minibatches 0 --verbose 0 --resume --train-json dump/tr_babel10_tr_babel10/deltafalse/data.json --valid-json dump/dt_babel10_tr_babel10/deltafalse/data.json --etype blstmp --elayers 4 --eunits 320 --eprojs 320 --subsample 1_2_2_1_1 --dlayers 1 --dunits 300 --atype location --aconv-chans 10 --aconv-filts 100 --mtlalpha 0.0 --batch-size 40 --maxlen-in 800 --maxlen-out 150 --sampling-probability 0.5 --opt adadelta --epochs 20 
EOF
) >exp/tr_babel10_pytorch_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_sampprob0.5_bs40_mli800_mlo150/train.log
time1=`date +"%s"`
 ( asr_train.py --ngpu 1 --backend pytorch --outdir exp/tr_babel10_pytorch_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_sampprob0.5_bs40_mli800_mlo150/results --debugmode 1 --dict data/lang_1char/train_units.txt --debugdir exp/tr_babel10_pytorch_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_sampprob0.5_bs40_mli800_mlo150 --minibatches 0 --verbose 0 --resume --train-json dump/tr_babel10_tr_babel10/deltafalse/data.json --valid-json dump/dt_babel10_tr_babel10/deltafalse/data.json --etype blstmp --elayers 4 --eunits 320 --eprojs 320 --subsample 1_2_2_1_1 --dlayers 1 --dunits 300 --atype location --aconv-chans 10 --aconv-filts 100 --mtlalpha 0.0 --batch-size 40 --maxlen-in 800 --maxlen-out 150 --sampling-probability 0.5 --opt adadelta --epochs 20  ) 2>>exp/tr_babel10_pytorch_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_sampprob0.5_bs40_mli800_mlo150/train.log >>exp/tr_babel10_pytorch_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_sampprob0.5_bs40_mli800_mlo150/train.log
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp/tr_babel10_pytorch_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_sampprob0.5_bs40_mli800_mlo150/train.log
echo '#' Finished at `date` with status $ret >>exp/tr_babel10_pytorch_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_sampprob0.5_bs40_mli800_mlo150/train.log
[ $ret -eq 137 ] && exit 100;
touch exp/tr_babel10_pytorch_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_sampprob0.5_bs40_mli800_mlo150/q/sync/done.15154
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o exp/tr_babel10_pytorch_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_sampprob0.5_bs40_mli800_mlo150/q/train.log -l 'hostname=b1[12345678]*|c*,gpu=1' -q g.q -l mem_free=2G,ram_free=2G   /export/a08/obask/espnet_cyc/egs/jsalt18e2e/asr1/exp/tr_babel10_pytorch_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_sampprob0.5_bs40_mli800_mlo150/q/train.sh >>exp/tr_babel10_pytorch_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_sampprob0.5_bs40_mli800_mlo150/q/train.log 2>&1
