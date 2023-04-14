# File Documentation
The documentation is not finished. There are some files (especially in the subdirectories) without documentation right now.
## Table of Contents
The documentation for the listed files is given below:
- [data_preparation.sh](#data_preparation)
- [make_files.py](#make_files)
- [pretrain.py](#pretrain)

---

### data_preparation.sh
**Short description:** Prepare Dataset basic structure script<br>
**Parameters:**

| Parameter Name | Function |
|----------------|----------|
| <code>sdir=$1</code> | source directory of the data |
| <code>dset=$2</code> | dataset part (Train, Test, Val, pretrain) |
| <code>segment=$3</code> | if do segmentation for pretrain set |
| <code>nj=$4 </code> | if multi cpu processing, default is true |

---

### make_files.py
**Short description:** Generate the text, utt2spk and wav.scp file<br>
**Parameters:**

| Parameter Name | Function |
|----------------|----------|
| <code>sys.argv[1]</code>, sourcedir (str) | The LRS2 dataset dir (e.g. /LRS2/data/lrs2_v1/mvlrs_v1/main) |
| <code>sys.argv[2]</code>, filelistdir (str) | The directory containing the dataset Filelists (METADATA) |
| <code>sys.argv[3]</code>, savedir (str) | Save directory, datadir of the clean audio dataset  |
| <code>sys.argv[4]</code>, dset (str) | Which set. There are pretrain, Train, Val, Test set |
| <code>sys.argv[5]</code>, nj (str) | Number of multi processes |


---

### pretrain.py
**Short description:** Prepare pretrain dataset<br>
**Parameters:**

| Parameter Name | Function |
|----------------|----------|
| <code>sys.argv[1]</code>, sourcedir (str) | The LRS2 dataset dir (e.g. /LRS2/data/lrs2_v1/mvlrs_v1/main) |
| <code>sys.argv[2]</code>, filelistdir (str) | The directory containing the dataset Filelists (METADATA) |
| <code>sys.argv[3]</code>, savedir (str) | Save directory, datadir of the clean audio dataset |
| <code>sys.argv[4]</code>, dset (str) | Which set. For this code dset is pretrain set |
| <code>sys.argv[5]</code>, nj (str) | Number of multi processes |
| <code>sys.argv[6]</code>, segment (str) |  If do segmentation |
