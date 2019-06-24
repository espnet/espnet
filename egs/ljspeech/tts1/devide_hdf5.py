#!/usr/bin/env python

# (Katsuki Inoue)
# 

import h5py
import os

def devide_hdf5(dir, file):
    in_file= os.path.join(dir, file)

    with h5py.File(in_file, "r") as f:
        for i, name in enumerate(f.keys()):
            out_name = name + ".h5"
            out_file = h5py.File(os.path.join(dir, out_name), "w")
            out_file.create_dataset("melspc", data = f[name].value)
            out_file.flush()
            out_file.close()

if __name__=='__main__':
    dir = "exp/sample_70-200/hdf5/eval" #"exp/train_no_dev_pytorch_taco2_r1_enc512-3x5x512-1x512_dec2x1024_pre2x256_post5x5x512_forward_ta128-15x32_cm_bn_cc_msk_pw1.0_do0.5_zo0.1_lr1e-3_ep1e-6_wd0.0_bs32_sd1/tmp/eval"
    
    for file in os.listdir(dir):
        head, tail = os.path.splitext(file)
        if ( tail == ".h5"):
            devide_hdf5(dir, file)
