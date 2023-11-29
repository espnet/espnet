# Kaldi steps

## License
The scripts in this directory were copied from Kaldi-ASR, https://github.com/kaldi-asr/kaldi, and their licenses follow the original license, https://github.com/kaldi-asr/kaldi/blob/master/COPYING, of Kaldi.

## Usage

To use the scripts under `steps`, **you need to install Kaldi**. See https://espnet.github.io/espnet/installation.html#step-1-optional-install-kaldi

### Extract fbank feats

```sh
number_of_parallel_jobs=32
cmd=utils/run.pl  # utils/slurm.pl, utils/queue.pl or utils/pbs.pl
steps/make_fbank_pitch.sh --nj "${number_of_parallel_jobs}" --cmd "${cmd}" data/train
```

You can find `data/train/feats.scp`.

The scirpts of Kaldi expect the directory specified by the argument, `data/train` in this case, following **a specific directory structure**. See https://github.com/espnet/data_example

See about `utils/run.pl`: https://espnet.github.io/espnet/parallelization.html
