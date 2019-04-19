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

To achieve 1 involves checking out a specific git commit that was used to train
the models. Start with:

	git checkout <some git hash>

You also need to download the CMU Wilderness data:

Git clone https://github.com/festvox/datasets-CMU_Wilderness

Follow the instructions there to download the data. Reference that download
point in your preprocess.sh script.

Then you need to run preprocess.sh. This will preprocess the audio data, create
the train/dev/eval splits, and create pronunciation lexicons.

Then you need to call run.sh. Give an overview of the command-line arguments.

Note to self:
- This README will need to be on the master branch of ESPnet, but the scripts
  etc that are specific to the old version should only be visible on those
  branches. This will mean, until scripts are tested with a new version, that
  there will only be a README.txt on the master branch.
