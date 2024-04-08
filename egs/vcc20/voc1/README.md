# Usage

This recipe trains a neural waveform generation module (vocoder) called [Parallel WaveGAN](https://arxiv.org/abs/1910.11480) (PWG) based on the [open source project](https://github.com/kan-bayashi/ParallelWaveGAN) developed by [kan-bayashi](https://github.com/kan-bayashi). It provides high-fidelity, faster-than-real-time synthesis, which is suitable for fast experiment cycles.

## Dataset preparation

The speakers available in each of tasks 1 and 2 are different, so we train different PWGs w.r.t. the task you participate in. Please specify `db_root` correspondingly, e.g. `db_root=../vc1_task1/downloads/official_v1.0_training`. Also, the list files that split train/dev sets should also be properly set, e.g. `list_dir=../vc1_task1/local/lists`.

## Execution

To train a PWG for task1, execute the main script as follows. (Please make sure `task` and `conf` is carefully set.)

```
$ ./run.sh --task task1 --conf conf/parallel_wavegan.task1.yaml
```

With this main script, a full procedure of PWG training is performed:

- Stage 0: Data preparation. A subset of speakers is chosen depending on the task.
- Stage 1: Feature extraction. This includes mel filterbanks extraction, stats computation, and feature normalization.
- Stage 2: Model training.
- Stage 3: Decoding. We decode the development set.

## Notes

The VCC2020 dataset is of 24 kHz. In task 1, since the LibriTTS dataset is also of 24kHz, we directly train a PWG that conditions on natural mel filterbanks extracted from 24kHz waveform samples to generate 24kHz waveform output. Such PWG would be referred to as a 24kHz-24kHz PWG. However, in task 2, since some datasets that are used for pretraining are of 16kHz, we train a 16kHz-24kHz PWG.
