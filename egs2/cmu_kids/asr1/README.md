# CMU Kids RECIPE

This is the recipe of the children speech recognition model with [CMU Kids dataset](https://catalog.ldc.upenn.edu/LDC97S63).

Before running the recipe, please download from https://catalog.ldc.upenn.edu/LDC97S63.
Then, edit 'CMU_KIDS' in `db.sh` and locate unzipped dataset as follows:

```bash
$ vim db.sh
CMU_KIDS=/path/to/cmu_kids 

$ tree -L 2 /path/to/cmu_kids
/path/to/cmu_kids
└── cmu_kids
    ├── doc
    ├── kids
    └── table
    └── 0readme.1st
```


# References
[1] Maxine S. Eskenazi; KIDS: A database of children’s speech. J. Acoust. Soc. Am. 1 October 1996; 100: 2759. https://doi.org/10.1121/1.416340