#!/usr/bin/env bash
set -euo pipefail

# Installing ffmpeg from conda-forge causes 2 problems:
#
# 1) ocl-icd-system will be installed, and ocl-icd_activate.sh will either
#       - Show "WARNING: No ICDs were found" when doing future conda install
#       - Fail due to [[ ]] being a bash feature, not standard shell syntax
#       See https://github.com/conda-forge/ocl-icd-feedstock/issues/29
#    We remove the script to correct this.
#
# 2) libarchive will be replaced by the conda-forge version. This breaks libmamba as libmamba needs libarchive.so.19
#       while the conda-forge version numbering is libarchive.so.13.6
#       See https://github.com/conda-forge/libarchive-feedstock/issues/69
#    We restore libarchive from the main channel to correct this.

libarchive=$(conda list | grep libarchive | awk 'BEGIN { OFS="="} { print $1,$2,$3 }')

conda install -y ffmpeg -c conda-forge
rm "${CONDA_PREFIX}/etc/conda/activate.d/ocl-icd_activate.sh" || true
conda install -y "${libarchive}" -c main --no-deps  # certifi should be included, but doesn't matter
