#!/usr/bin/env python3
#  coding: utf-8

import argparse
from kaldiio import ReadHelper
import numpy as np
import os
from os.path import join
import sys
import h5py

def write_hdf5(hdf5_name, hdf5_path, write_data, is_overwrite=True):
    """FUNCTION TO WRITE DATASET TO HDF5

    Args :
        hdf5_name (str): hdf5 dataset filename
        hdf5_path (str): dataset path in hdf5
        write_data (ndarray): data to write
        is_overwrite (bool): flag to decide whether to overwrite dataset
    """
    # convert to numpy array
    write_data = np.array(write_data)

    # check folder existence
    folder_name, _ = os.path.split(hdf5_name)
    if not os.path.exists(folder_name) and len(folder_name) != 0:
        os.makedirs(folder_name)

    # check hdf5 existence
    if os.path.exists(hdf5_name):
        # if already exists, open with r+ mode
        hdf5_file = h5py.File(hdf5_name, "r+")
        # check dataset existence
        if hdf5_path in hdf5_file:
            if is_overwrite:
                logging.warning("dataset in hdf5 file already exists.")
                logging.warning("recreate dataset in hdf5.")
                hdf5_file.__delitem__(hdf5_path)
            else:
                logging.error("dataset in hdf5 file already exists.")
                logging.error("if you want to overwrite, please set is_overwrite = True.")
                hdf5_file.close()
                sys.exit(1)
    else:
        # if not exists, open with w mode
        hdf5_file = h5py.File(hdf5_name, "w")

    # write data to hdf5
    hdf5_file.create_dataset(hdf5_path, data=write_data)
    hdf5_file.flush()
    hdf5_file.close()

def get_parser():
    parser = argparse.ArgumentParser(
        description='Convet kaldi-style features to h5 files for WaveNet vocoder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scp_file', type=str, help='scp file')
    parser.add_argument('--out_dir', type=str, help='output directory')
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])
    os.makedirs(args.out_dir, exist_ok=True)
    with ReadHelper(f"scp:{args.scp_file}") as f:
        for utt_id, arr in f:
            out_path = join(args.out_dir, "{}-feats.h5".format(utt_id))
            write_hdf5(out_path, "/melspc", np.float32(arr))
