# ESPnet2
We are planning super major update, called `ESPnet2`.

The developing status is still **under construction** yet, so please be very careful to  use with understanding following cautions:

- There might be fatal bugs related to essential parts.
- We haven't achieved comparable results to espnet1 on each tasks yet.

## Main changing from ESPnet1

- **Chainer free**
  - Discarding Chainer completely.
  - The development of Chainer is stopped at v7: https://chainer.org/announcement/2019/12/05/released-v7.html
- **Kaldi free**
  - It's not mandatory to compile Kaldi.
      - **If you find some recipes requiring Kaldi mandatory, please report it. It should be dealt as a bug in ESPnet2.**
  - We still support the features made by Kaldi optionally.
  - We still follow Kaldi style. i.e. depending on `utils/` of Kaldi.
- **On the fly** feature extraction & text preprocessing for training
  - You don't need to create the feature file before training, but just input wave data directly.
      - We support both raw wave input and extracted feature.
  - The preprocessing for text, tokenization to characters, or sentencepieces, can be also applied during training.
- Discarding JSON format for describing training corpus in espnet1.
    - Why do we discard the JSON format? Because a dict object generated from a large JSON file requires much memory and it also takes much time to parse such a large JSON file.
- Support distributed data-parallel training (Not enough tested)
   - Single node multi GPU training with `DistributedDataParallel` is also supported.

## Recipes using ESPnet2

You can find the new recipes in `egs2`:

```
espnet/   # Python modules of epsnet1
espnet2/  # Python modules of epsnet2
egs/      # espnet1 recipes
egs2/     # espnet2 recipes
```

The usage of recipes is **almost same** as that of ESPnet1.


1. Change directory to the base directory

    ```bash
    # e.g.
    cd egs2/an4/asr1/
    ```

    `an4` is tiny corpus and can be freely obtained, so it might be suitable for this tutorial.

    Note that all shell scripts are supposed to be executed in the same level directory, `egs2/*/asr1`, and they can't be used in other place.

    ```bash
    # Doesn't work
    cd egs2/an4/
    ./asr1/run.sh
    ```

    ```bash
    # Doesn't work
    cd egs2/an4/asr1/local/
    ./data.sh
    ```

1. Change the configuration

    ```
    egs2/an4/asr1/
      - conf/      # Configuration files for training, inference, etc.
      - scripts/   # Bash utilities of espnet2
      - pyscripts/ # Python utilities of espnet2
      - steps/     # Kaldi utilities
      - utils/     # Kaldi utilities
      - db.sh      # The directory path of each corpora
      - path.sh    # Setup script for environment variables
      - cmd.sh     # Configuration for your backend of job scheduler
      - run.sh     # Entry point
      - asr.sh     # Invoked by run.sh
    ```

    - You need to modify `db.sh` for specifying your corpus before executing `run.sh`. For example, you'll the recipe of `egs2/wsj`, you need to change the value of `WSJ0` and `WSJ1`.
        - Some corpus can be freely obtained from WEB and they are written as "downloads/" at the initial state. You can also change them to your corpus path if it's already downloaded.
    - `path.sh` is used to set up the environment for `run.sh`. Note that the Python interpreter used for ESPnet is not the current Python of your terminal, but it's the Python which was installed at `tools/`. Thus you need to source `path.sh` to use this Python.
    ```bash
    source path.sh
    python
    ```
    - `cmd.sh` is used for specifying the backend of JOB scheduler. If you don't have such system in your machine environment, you don't need to change anything about this file. See [Using Job scheduling system](./parallelization.md)

1. Run `run.sh`

    ```bash
    ./run.sh
    ```

    `run.sh` is an example script, which we often call as "recipe", to run all stages related to DNN experiments; data-preparation, training, and evaluation.

## See training status

### Log file

```bash
less exp/asr_train_<some-name>/train.log
```

### Show images

```bash
# Accuracy plot
# (eog is Eye of GNOME Image Viewer)
eog exp/asr_train_<some-name>/images/acc.img
# Attention plot
eog exp/asr_train_<some-name>/<sample-id>/<param-name>.img
```

### Use tensorboard

```bash
tensorboard --logdir exp/asr_train_<some-name>/tensorboard/
```

# Instruction for run.sh
We use all python commands via `run.sh` and you may need to invoke a Python script directly in some cases, but `run.sh` itself is configurable and you can live comfortably with it in many cases without any modifications.

## How to parse command line arguments in shell scripts?

All shell script in espnet/espnet2 depends on [utils/parse_options.sh](https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/utils/parse_options.sh) to parase command line arguments.

e.g. If the script has `ngpu` option

```bash
#!/bin/bash
# run.sh
ngpu=1
. utils/parse_options.sh
echo ${ngpu}
```

Then you can change the value as following:

```bash
$ ./run.sh --ngpu 2
echo 2
```

You can also show the help message if it supports.

```bash
./run.sh --help
```


## Start from specified stage and stop at specified stage
The procedures in `run.sh` can be divided into some stages, e.g. data preparation, training, and evaluation. You can specify the starting stage and the stopping stage.

```bash
./run.sh --stage 2 --stop-stage 6
```

## Change the configuration for training
See  about the usage of training tool.

(The following is the case of ASR training and you need to replace it accordingly)

```bash
# Give a configuration file
./run.sh --asr_train_config conf/train_asr.yaml
# Give arguments to training tool directly
./run.sh --asr_args "--foo arg --bar arg2"
```
