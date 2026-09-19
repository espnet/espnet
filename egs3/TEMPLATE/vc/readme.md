# ESPnet3 VC recipe

Shared defaults for voice-conversion recipes. The only method shipped so far is
**kNN-VC** (Baas, van Niekerk and Kamper, Interspeech 2023,
<https://github.com/bshall/knn-vc>), integrated into ESPnet3 in
`egs3/librispeech_100/vc`; that recipe's readme carries the acknowledgement,
license notes and citation. A concrete recipe only overrides what differs from
the configs in `conf/`.

## Quick start

```bash
# 0) Edit configs to set paths. When `--training_config` is also passed to
#    `infer` or `measure`, run.py propagates only the experiment path fields
#    (`exp_tag`/`exp_dir`); anything else an inference config interpolates,
#    such as `wavlm_checkpoint`, must be defined in that config too.

# 1) Check the corpus is available (raw-passthrough builder)
python run.py --stages create_dataset --training_config conf/training.yaml

# 2) Precompute (prematched) WavLM features for the vocoder training set
python run.py --stages prepare_features --training_config conf/training.yaml

# 3) Train the HiFi-GAN vocoder on those features
python run.py --stages train --training_config conf/training.yaml

# 4) Convert the test pairs (source utterance -> target speaker) to WAV files
python run.py --stages infer \
    --training_config conf/training.yaml --inference_config conf/inference.yaml
```
