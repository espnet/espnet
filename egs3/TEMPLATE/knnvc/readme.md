# ESPnet3 VC recipe

Shared defaults for voice-conversion recipes. The only method shipped so far is
**kNN-VC** (Baas, van Niekerk and Kamper, Interspeech 2023,
<https://github.com/bshall/knn-vc>), integrated into ESPnet3 in
`egs3/librispeech_100/knnvc`; that recipe's readme carries the acknowledgement,
license notes and citation. A concrete recipe only overrides what differs from
the configs in `conf/`.

## Quick start

```bash
# 0) Edit configs to set paths. A recipe's run.py is `launch(KNNVCSystem)`;
#    its stages are the system's declared ones (create_dataset,
#    prepare_features, train, infer, measure, pack_model, upload_model). When
#    `--training_config` is also passed to `infer` or `measure`, launch()
#    propagates only the experiment path fields (`exp_tag`/`exp_dir`);
#    anything else an inference config interpolates, such as
#    `wavlm_checkpoint`, must be defined in that config too.

# 1) Check the corpus is available (raw-passthrough builder)
python run.py --stages create_dataset --training_config conf/training.yaml

# 2) Precompute (prematched) WavLM features for the vocoder training set
python run.py --stages prepare_features --training_config conf/training.yaml

# 3) Train the HiFi-GAN vocoder on those features
python run.py --stages train --training_config conf/training.yaml

# 4) Convert the test pairs (source utterance -> target speaker) to WAV files
python run.py --stages infer \
    --training_config conf/training.yaml --inference_config conf/inference.yaml

# 5) Score them: ASR WER/CER against the source transcript, read from the data
python run.py --stages measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml --metrics_config conf/metrics.yaml
```

## The inference boundary

`conf/inference.yaml` declares the contract every recipe, the published
bundle and a demo share: `speech` (16 kHz mono `audio`), `reference_speech`
(`reference_audio`, a non-empty list of such utterances) and `target_speaker`
(`text`) in, `wav` (16 kHz mono `audio`) out, keyed by the conversion's
`pair_id`. `reference_audio` and the vocoder view's `features` kind are
defined in `src/contracts.py`, which the recipe datasets import and the
contract loads through `contract.imports`; a published bundle therefore
needs `trust_user_code=True`. The demo stages are not declared: the demo UI
has no asset type for a set of reference utterances yet.
