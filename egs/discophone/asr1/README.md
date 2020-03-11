# BABEL RECIPE

## Setting Up Experiments
To setup an experiment (using one or more languages in training) simply run
from this directory ...

`./setup_experiment.sh <expname>`

This will copy all the necessary files for running an experiment to the created
directory ../expname

Possibly the most important of these files is 

`conf/lang.conf`

This file is a giant configurations file storing all the necessary data paths
for all the BABEL languages. The entry for one language looks like the
following:

~~~~
# Pashto
train_data_dir_104=/export/babel/data/104-pashto/release-current/conversational/training
train_data_list_104=./conf/lists/104-pashto/train.LimitedLP.list
train_data_dir_104_FLP=/export/babel/data/104-pashto/release-current/conversational/training
train_data_list_104_FLP=./conf/lists/104-pashto/training.list
dev10h_data_dir_104=/export/babel/data/104-pashto/release-current/conversational/dev
dev10h_data_list_104=./conf/lists/104-pashto/dev.list
lexicon_file_104=/export/babel/data/104-pashto/release-current/conversational/reference_materials/lexicon.txt
lexiconFlags_104="--romanized --oov <unk>"
~~~~

You'll notice there are two kinds of paths specified:
  1. ./conf.*
  2. /export/*

### ./conf/*
All paths of the first variety are predifined data splits from the kaldi babel
recipe (See, ../../../tools/kaldi/egs/babel/s5d for more info.). In the babel
data there are a few datasets that are commonly used. The first is original
training set. It is generally about 40h of data, but depending on the language
it can be as high as 80h. It is called the Full-Language-Pack, FLP, or FullLP.
The second dataset is a subset of the FLP. It is a 10h training set called the
Limited-Language-Pack, LLP, or LimitedLP. The last set is a 10h testset 
generally called the dev10h.

You can specify which training set to use with the --FLP [true/false] flag,
which set a variable $FLP with default value of true.

### /export/*
The second type of path references the location of the data your local machine
and will hencefor each language you use in training you will have to set these 
to the correct paths for your setup (Depending on where your LDC distribution
resides). The path to the lexicon file is required to determine the vocabulary,
but the pronunciations are not used. Some lexicons for languages written in
non-roman scripts come with transliterations. In order to properly parse these
files we must provide the appropriate flags which is accomplished by setting
the variable lexiconFlags_*.


## Running Experiments
To run the experiment do 

`cd ../expname`

To specify the BABEL langauges in training refer to them by their language id.
See  conf/lang.conf for the exhaustive list of languages and corresponding
language ids.

### Examples
`./run.sh --langs "102" --recog "102"`
  This will create a network trained on 102 (assamese) and will test it on 102.

`./run.sh --langs "102 103" --recog " 102 103"`
  This will create a network trained on 102 and 103 (bengali) and tests on both.

## Training / Decoding with RNNLM

The transcript can be used to train a grapheme level language model.
It looks for the created train directory and trains a language model on the 
data/train/text file, for instance present in the path specified by 

data/${train_set} (see run.sh)

If a byte-pair encoded version of this file is provided in this path than the
RNNLM can be trained on byte-pairs (word-pieces).

