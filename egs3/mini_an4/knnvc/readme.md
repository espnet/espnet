# mini_an4 kNN-VC recipe

The smallest recipe that exercises every kNN-VC stage. It exists for CI and for
checking a change end to end in seconds; it is **not** a voice-conversion
system, and the audio it produces is not a conversion of anything.

Two substitutions make it fast enough to run on CPU with no download:

- the encoder is `src/stub_encoder.py`, a fixed random projection standing in
  for WavLM-Large (1.2 GB) with the same interface;
- the HiFi-GAN is deliberately tiny (16 channels, two upsampling stages).

The corpus is the an4 sample that ships with the repository;
`downloads.tar.gz` is a symbolic link to the copy in `egs3/mini_an4/asr`.

For the real thing, see `egs3/librispeech_100/knnvc`.

## Quick start

```bash
python run.py --stages create_dataset prepare_features train infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml
```

That is the same sequence `ci/test_integration_espnet3.sh` runs, and it takes
about ten seconds.

## What the stages do here

| Stage | |
|---|---|
| `create_dataset` | extracts the bundled archive into `downloads/` |
| `prepare_features` | encodes 6 utterances, prematched within each of 3 speaker pools |
| `train` | 2 batches through the GAN path, both optimizers stepping |
| `infer` | converts 2 source/target-speaker pairs to WAV |

One an4 speaker has a single utterance, so the run also covers the case where a
prematching pool has nothing to match against and the features are written
unmatched.
