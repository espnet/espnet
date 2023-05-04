This recipe trains a multilingual model using data from the CMU Wilderness
Multilingual Speech Dataset (Black et al., 2019).

There are going to be two forms of this recipe:
	1. Almost exactly similar setup to that used by Adams et al. (2019). Use
	this to replicate the results of that paper. This involves checking out and
	using an older version of ESPnet.
	2. A recipe that trains using the latest
	version of ESPNet. Use this if you don't care about replication: You just
	want a multilingual system that uses the CMU Wilderness data. This is still
	to be implemented. It should be pretty straightforward to adapt the recipe
	for (1) to work with the latest ESPnet though.

===1. Replicating Adams et al. 2019===

First we need to checking out a specific git commit that restores the code to
the state it was in when the models were trained. Start with:

	git checkout <some git hash>

You may want to make sure Kaldi and the right python virtual environment is
installed for the now-checked-out version of Kaldi, since things might have
changed since we ran experiments. See ../../../README.md for instructions.

You also need to download the CMU Wilderness data:

	git clone https://github.com/festvox/datasets-CMU_Wilderness

Follow the instructions in that repository to download the data for each of the
languages you want to train or test on. For lists of relevant language codes,
look in egs/cmu_wilderness/fixed-phoneme/conf/langs/. If you want to train a
100 language model, use the languages in
egs/cmu_wilderness/fixed-phoneme/conf/langs/ninetynine (named so because
there's actually 99 pretraining sets; adaptation adds another one to make 100).
For the geographically similar or phonologically/phonetically similar language
lists, look at the file <language iso 639-3 code>-geo or <language iso 639-3
code>-phon+inv in the same directory. Set the path to that repository to your
$datasets variable in the egs/cmu_wilderness/fixed-phoneme/preprocess.sh
script.

Now we need to do some data preprocessing. First, change directory to
egs/cmu_wilderness/fixed-phoneme/. Comment/uncomment lines as needed in cmd.sh
depending on your system. Then run ./preprocess.sh. This will preprocess the
audio data, create the train/dev/eval splits, and create pronunciation
lexicons. Simply call ./preprocess.sh as is to preprocess all the data. As
alluded to above, various collections of CMU Wilderness language codes are kept
in conf/langs/. If you want to move beyond replication and use training
languages of your choice, create a
list in conf/langs/ with any name and list the reading codes in it in a fashion
similar to the other files. Then call ./preprocess --train-groups
conf/langs/<your-list-name>.

Then to train you need to call run.sh. Here's an overview of the arguments that
organically grew out of experimentation. They (and the whole pipeline, and
probably this README) could be clearer; I apologize.

To train a 100-language model with only the grapheme objective, you'd go:

./run.sh --train-langs ninetynine

To train the same model with the phoneme objective, go:

./run.sh --train-langs ninetynine --mtlalpha 0.33 --phoneme-objective-weight 0.33 --stage 3

The stage 3 just skips some preprocessing of stages 1 and 2 which were already
done by the previous command (which uses the same training set). --mtlalpha is
a standard ESPnet argument that weights CTC grapheme training against the
attentional model. --phoneme-objective-weight throws in a phoneme CTC objective
into the mix. Whatever weight is left over after --mtlalpha and
--phoneme-objective-weight goes to the attentional decoder.

To add the language-adversarial objective, go:

./run.sh --train-langs ninetynine --mtlalpha 0.33 --phoneme-objective-weight 0.33 --predict-lang adv --predict-lang-alpha-scheduler ganin

"ganin" can be replaced with "shinohara", which in our case didn't usually do
as well. This variable just determines how the learning rate for the
adversarial objective gets scheduled. "ganin" is so named because it follows
the schedule of Ganin et al. (2016). "shinohara" follows Shinohara et al.
(2016).

=Adaptation=

Once pretraining is complete, you can adapt your model to the target language.
To do this you call the same command you used to train, but with an
additional --adapt-langs argument that says what language you're adapting to,
as well as a --recog-set argument that gives the set you want to test on. We
recommend you also set --adapt-no-phoneme true, which drops the phoneme
objective in adaptation and tends to yield better results.

So to adapt that first model to a reading of South Bolivian Quechua (see the
relationship between codes and languages here:
http://festvox.org/cmu_wilderness/), you could go:

./run.sh --train-langs ninetynine --adapt-langs QUHRBV --recog-set QUHRBV_eval

===2. Training a CMU Wilderness model with the latest version of ESPnet.===

raise NotImplementedError("Not yet implemented.")

===References===

Oliver Adams, Matthew Wiesner, Shinji Watanabe, David Yarowsky. 2019. Massively
Multilingual Adversarial Speech Recognition. In NAACL.

Alan W Black. 2019. CMU Wilderness Multilingual Speech Dataset. In ICASSP.
UK. https://github.com/festvox/datasets-CMU_Wilderness

Yaroslav  Ganin,  Evgeniya  Ustinova,  Hana  Ajakan, Pascal  Germain,  Hugo
Larochelle,  Francois  Laviolette, Mario  Marchand, Victor Lempitsky. 2016.
Domain-adversarial training of neural networks. Journal of Machine Learning
Research,17(1).

Yusuke Shinohara. 2016. Adversarial multi-task learning of deep neural networks
for robust speech recognition. In INTERSPEECH.
