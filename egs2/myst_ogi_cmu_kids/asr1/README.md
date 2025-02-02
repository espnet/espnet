# MyST + OGI Kids + CMU Kids RECIPE

This is the recipe of the children speech recognition model with [MyST dataset](https://catalog.ldc.upenn.edu/LDC2021S05), [CSLU: OGI Kids dataset](https://catalog.ldc.upenn.edu/LDC2007S18), and [CMU Kids dataset](https://catalog.ldc.upenn.edu/LDC97S63).

Before running the recipe, please download from https://catalog.ldc.upenn.edu/LDC2021S05, https://catalog.ldc.upenn.edu/LDC2007S18, https://catalog.ldc.upenn.edu/LDC97S63.
Then, edit 'MYST', 'OGI_KIDS', and 'CMU_KIDS' in `db.sh` and locate unzipped dataset as follows:

```bash
$ vim db.sh
MYST=/path/to/myst
OGI_KIDS=/path/to/cslu_kids
CMU_KIDS=/path/to/cmu_kids

$ tree -L 2 /path/to/myst
/path/to/myst
└── myst_child_conv_speech
    ├── data
    ├── docs
    └── index.html

$ tree -L 1 /path/to/cslu_kids
/path/to/cslu_kids
├── docs
├── index.html
├── labels
├── misc
├── speech
├── trans
└── verify

$ tree -L 1 /path/to/cmu_kids
/path/to/cmu_kids
├── 0readme.1st
├── doc
├── kids
└── tables
```
