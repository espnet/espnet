# JIBO Kids RECIPE

This is the recipe of the children speech recognition model with [JIBO Kids dataset](https://github.com/balaji1312/Jibo_Kids).

Before running the recipe, please download the [dataset](https://github.com/balaji1312/Jibo_Kids).
Then, edit 'JIBO_KIDS' in `db.sh` and locate unzipped dataset as follows:

```bash
$ vim db.sh
JIBO_KIDS=/path/to/jibo_kids 

$ tree -L 2 path/to/jibo_kids 
/path/to/jibo_kids
├── data
│   ├── blocks
│   ├── brush
│   ├── colors
│   └── letters_digits
└── README.txt

```

## References

[1] Shankar, Natarajan Balaji, et al. "The JIBO Kids Corpus: A speech dataset of child-robot interactions in a classroom environment." JASA Express Letters 4.11 (2024).
