This recipe trains a multilingual model using data from the CMU Wilderness
Multilingual Speech Dataset (Black et al., 2019).

There are two forms of this recipe:
1. Almost exactly similar setup to that used by Adams et al., 2019. Use this to
replicate the results of that paper. This involves using an older version of
ESPnet.
2. A recipe that trains using the latest version of ESPNet. Use this if you
don't care about replication: You just want a multilingual system that uses the
CMU Wilderness data. This is still to be implemented. It should be pretty
straightforward to adapt the recipe for 1 to work with the latest ESPnet
though.

===1. Replicating Adams et al. 2019===

To achieve 1 involves checking out a specific git commit that was used to train
the models. Start with:

	git checkout <some git hash>

You may want to make sure Kaldi and the right python venv is installed for the
now-checked-out version of Kaldi, since things might have changed since we ran
experiments. See ../../../README.md.

You also need to download the CMU Wilderness data:

	git clone https://github.com/festvox/datasets-CMU_Wilderness

Follow the instructions in that repository to download the data for each of the
languages you want to train/test on. Use a Bash command to loop over all the
langauges you're interested in. <Include said command, as well as a list of all
the relevant langauges>. Set the path to that repository to your datasets
variable in egs/cmu_wilderness/fixed-phoneme/preprocess.sh script.

Then you need to run preprocess.sh. This will preprocess the audio data, create
the train/dev/eval splits, and create pronunciation lexicons. Simply call
./preprocess.sh as is to preprocess all the data. Various collections of
CMU-W language codes are kept in conf/langs/. If you want to create your own list of
languages here, create a list in conf/langs/ with any name. Then call
./preprocess --train-groups conf/langs/<your-list-name>.

Then you need to call run.sh. Give an overview of the command-line arguments. I
apologize for these arguments. Making another recipe is on the to-do list.

To train a 100-language model with only the grapheme objective, you'd go:

./run.sh --train-langs ninetynine

To train the same model with the phoneme objective, go:

./run.sh --train-langs ninetynine --mtlalpha 0.33 --phoneme-objective-weight 0.33

To add the language-adversarial objective, go:

./run.sh --train-langs ninetynine --mtlalpha 0.33 --phoneme-objective-weight
0.33 --predict-lang adv --predict-lang-alpha-scheduler <ganin|shinohara>

=Adaptation=

Once pretraining is complete, you can adapt your model. To do this you
basically call the same command you used to train, but with an additional
--adapt-langs argument that says what language you're adapting to, and
--recog-set gives the set you want to test on. We recommend you also set
--adapt-no-phoneme true, which drops the phoneme objective in adaptation and
tends to yield better results.

Why 99? Train langs have 99 languages, when you adapt you get your 100th :).

Note to self:
- This README will need to be on the master branch of ESPnet, but the scripts
  etc that are specific to the old version should only be visible on those
  branches. This will mean, until scripts are tested with a new version, that
  there will only be a README.txt on the master branch.

===2. Training a CMU Wilderness model with the latest version of ESPnet.===

Work in progress.

===References===

Oliver Adams, Matthew Wiesner, Shinji Watanabe, David Yarowsky. 2019. Massively
Multilingual Adversarial Speech Recognition. In NAACL.

Alan W Black. 2019. CMU Wilderness Multilingual Speech Dataset. In ICASSP.
UK. https://github.com/festvox/datasets-CMU_Wilderness


