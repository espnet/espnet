from __future__ import print_function
from __future__ import division
import numpy as np
import h5py
import sys
import os
import argparse


def write_hdf5(hdf5_name, hdf5_path, write_data, is_overwrite=True):
    """FUNCTION TO WRITE DATASET TO HDF5

    :param str hdf5_name: hdf5 format filename
    :param str hdf5_path: dataset path in hdf5
    :param array write_data: numpy array
    :param bool is_overwrite: flag to decide whether to overwrite dataset
    """
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
                print("WARNING: data in hdf5 file already exists. recreate dataset in hdf5.")
                hdf5_file.__delitem__(hdf5_path)
            else:
                print("ERROR: there is already dataset.")
                print("if you want to overwrite, please set is_overwrite = True.")
                hdf5_file.close()
                sys.exit(1)
    else:
        # if not exists, open with w mode
        hdf5_file = h5py.File(hdf5_name, "w")

    # write data to hdf5
    hdf5_file.create_dataset(hdf5_path, data=write_data)
    hdf5_file.flush()
    hdf5_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stdin', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin, help='Output hdf5 filename')
    parser.add_argument('--write_name', '-w', type=str, required=True,
                        help='hdf5 filename to write')
    args = parser.parse_args()

    # write to hdf5 dataset
    for line in args.stdin:
        if "[" in line:
            utt_id = line.split(" ")[0]
            feat = []
        elif "]" in line:
            feat.append(np.array(line[:-2].split(), dtype=np.float32))
            feat = np.stack(feat, axis=0)
            write_hdf5(args.write_name, utt_id, feat)
        else:
            feat.append(np.array(line.split(), dtype=np.float32))

    # show info
    with h5py.File(args.write_name, "r") as f:
        utt_ids = f.keys()
        print("Converted %d feature metrices." % len(utt_ids))
