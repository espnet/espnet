# -*- coding: utf-8 -*-

# Copied from DNS-Challenge official repo file:
# https://github.com/microsoft/DNS-Challenge/edit/master/utils.py

# Original License: CC-BY 4.0 International:
# https://github.com/microsoft/DNS-Challenge/blob/master/LICENSE

# Retrieved on Sep. 7th, 2022, by Shih-Lun Wu (summer7sean@gmail.com)

"""
Created on Fri Nov  1 10:28:41 2019

@author: rocheng
"""
import csv
import glob
import os
from shutil import copyfile


def get_dir(cfg, param_name, new_dir_name):
    """Helper function to retrieve directory name if it exists,
    create it if it doesn't exist"""

    if param_name in cfg:
        dir_name = cfg[param_name]
    else:
        dir_name = os.path.join(os.path.dirname(__file__), new_dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def write_log_file(log_dir, log_filename, data):
    """Helper function to write log file"""
    data = zip(*data)
    with open(os.path.join(log_dir, log_filename), mode="w", newline="") as csvfile:
        csvwriter = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for row in data:
            csvwriter.writerow([row])


def str2bool(string):
    return string.lower() in ("yes", "true", "t", "1")


def rename_copyfile(src_path, dest_dir, prefix="", ext="*.wav"):
    srcfiles = glob.glob(f"{src_path}/" + ext)
    for i in range(len(srcfiles)):
        dest_path = os.path.join(dest_dir, prefix + "_" + os.path.basename(srcfiles[i]))
        copyfile(srcfiles[i], dest_path)
