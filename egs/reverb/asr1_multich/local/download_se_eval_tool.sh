#!/bin/bash
# Copyright 2018 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Copyright 2019 Johns Hopkins University (Author: Xiaofei Wang)
# This script downloads the official REVERB challenge SE scripts and SRMR toolbox
# This script also downloads and compiles PESQ
# please make sure that you or your institution have the license to report PESQ
# Apache 2.0

wget 'https://www.itu.int/rec/dologin_pub.asp?lang=e&id=T-REC-P.862-200102-I!!SOFT-ZST-E&type=items' -O PESQ.zip
unzip PESQ.zip -d local/PESQ_sources
rm PESQ.zip
cd local/PESQ_sources/P862/Software/source
gcc  *.c -lm -o PESQ
cd ../../../../../
mv local/PESQ_sources/P862/Software/source/PESQ local/

wget 'https://reverb2014.dereverberation.com/tools/REVERB-SPEENHA.Release04Oct.zip' -O REVERB_scores.zip
unzip REVERB_scores.zip -d local/REVERB_scores_source
rm REVERB_scores.zip

# Get the evaluation list
wget 'https://reverb2014.dereverberation.org/tools/taskFiles_et.tgz' -O taskFiles_et.tgz
tar -zxvf taskFiles_et.tgz -C local/REVERB_scores_source/REVERB-SPEENHA.Release04Oct/
cp -a local/REVERB_scores_source/REVERB-SPEENHA.Release04Oct/taskFiles_et/* local/REVERB_scores_source/REVERB-SPEENHA.Release04Oct/taskfiles
rm taskFiles_et.tgz

pushd local/REVERB_scores_source/REVERB-SPEENHA.Release04Oct/evaltools
sed -i 's/wavread/audioread/g' prog/score_sim.m
git clone https://github.com/MuSAELab/SRMRToolbox.git
sed -i 's/wavread/audioread/g' SRMRToolbox/libs/preprocess.m
sed -i 's/SRMR_main/SRMR/g' prog/score_real.m
sed -i 's/SRMR_main/SRMR/g' prog/score_sim.m
sed -i 's/+wb\ //g' prog/calcpesq.m
sed -i 's/pesq_/_pesq_/g' prog/calcpesq.m
sed -ie '30d;31d' prog/calcpesq.m
patch score_RealData.m -i ../../../score_RealData.patch -o score_RealData_new.m
mv score_RealData_new.m score_RealData.m
patch score_SimData.m -i ../../../score_SimData.patch -o score_SimData_new.m
mv score_SimData_new.m score_SimData.m
sed -i 's/dt/et/g' score_SimData.m
sed -i 's/dt/et/g' score_RealData.m
sed -i 's/Dev/Eval/g' score_RealData.m
popd

cp -a local/evaltools local/REVERB_scores_source/REVERB-SPEENHA.Release04Oct

