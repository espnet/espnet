# LibriSpeech 100h kNN-VC recipe

This recipe is an **ESPnet3 integration of kNN-VC** ("Voice Conversion With
Just Nearest Neighbors", Matthew Baas, Benjamin van Niekerk and Herman Kamper,
Interspeech 2023). The method, the model design, the training procedure and the
released checkpoints are the original authors' work; this recipe only wires
them into the ESPnet3 stage pipeline. See the acknowledgement and citation at
the end of this file.

Any-to-any voice conversion with kNN-VC: frozen WavLM-Large layer-6 features,
k-nearest-neighbours regression as the converter, and a HiFi-GAN vocoder
trained on *prematched* features of `train-clean-100`. Only the vocoder is
trained.

Place the corpus under `download/LibriSpeech`, or set the `LIBRISPEECH`
environment variable to an existing LibriSpeech root (`train-clean-100` and
`dev-clean` are required).

## Quick start

```bash
# 0) Check the corpus is available
python run.py --stages create_dataset --training_config conf/training.yaml

# 1) Precompute prematched WavLM features for train-clean-100 and dev-clean
#    (~38 GB of float16 features; GPU strongly recommended)
python run.py --stages prepare_features --training_config conf/training.yaml

# 2) Train the HiFi-GAN vocoder (the paper uses 2.5M updates)
python run.py --stages train --training_config conf/training.yaml

# 3) Convert 200 dev-clean utterances to other dev-clean speakers
python run.py --stages infer \
    --training_config conf/training.yaml --inference_config conf/inference.yaml
```

## Inference with the authors' checkpoints (no training)

```bash
python run.py --stages infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference_pretrained.yaml
```

Converted audio is written to
`exp/training/inference_pretrained/dev-clean/wav/<source_utt>_to_<target_spk>.wav`
and listed in `wav.scp`; `ref.scp` keeps the source transcripts for ASR-based
scoring.

The authors released two generators and both load here without conversion
work on your part:

| `vocoder_checkpoint` | Trained on | |
|---|---|---|
| `prematch_g_02500000.pt` | prematched features | their default, the paper's main system |
| `g_02500000.pt` | plain WavLM features | the non-prematched baseline |

`conf/inference_pretrained.yaml` uses the prematched one; point
`vocoder_checkpoint` at `g_02500000.pt` in the same release to convert with the
other and see what prematching contributes. The `*_do_*.pt` assets alongside
them are discriminator and optimizer states for resuming training at step
2500000, not inference checkpoints.

## Notes

- `wavlm_checkpoint` (a TEMPLATE default in both `training.yaml` and
  `inference*.yaml`) accepts a local path; by default the WavLM-Large checkpoint
  is fetched from the kNN-VC GitHub release. Override it in both files when
  using a local copy, since `run.py` does not propagate it between configs.
- `create_dataset` needs `train-clean-100` and `dev-clean`; `infer` alone only
  needs the split it converts, so the pretrained-checkpoint workflow above runs
  with `dev-clean` only. Point `create_dataset.source_dir` at the corpus root
  when it is neither under `download/` nor in `$LIBRISPEECH`.
- `prepare_features` honours `parallel:` in the training config and can be
  resumed; finished shards are skipped. On one RTX A6000, `dev-clean`
  (5.4 h, 2703 utterances) takes about 8 minutes including prematching.
- The vocoder trains on all of `train-clean-100` and validates on all of
  `dev-clean`, which is exactly the authors' `data_splits/*.csv`. A
  deterministic utterance-level holdout of `train-clean-100` is available
  instead via `subset: train` / `subset: valid` (+ `valid_ratio`).
- As in the official implementation, reference utterances are silence-trimmed
  with torchaudio's VAD (`vad_trigger_level: 7`, set `0` to disable) and the
  converted waveform is normalized to `tgt_loudness_db` (-16 LUFS). With these
  defaults the espnet3 conversions are numerically identical to the official
  `bshall/knn-vc` code for the released checkpoints.

## Relation to the original recipe

Kept identical to the official `bshall/knn-vc` code (verified numerically for
the released checkpoints, see the notes above):

- WavLM-Large layer 6 as matching and synthesis features, no waveform
  normalization, hop-aligned padding when preparing vocoder features;
- prematching with `k=4`, cosine distance, pooling the *other utterances of
  the same speaker chapter directory* (`prematch_pool: chapter`), which is what
  the official `prematch_dataset.py` does via `path.parent.rglob` even though
  the paper says "same speaker"; `prematch_pool: speaker` gives the paper's
  description;
- HiFi-GAN V1 generator with the `Linear(1024, 512)` input projection
  (`hifigan/config_v1_wavlm.json`), MSD + MPD discriminators, LSGAN losses,
  feature matching x2, L1 log-mel loss x45 with the same log-mel front end;
- AdamW (lr 2e-4, betas 0.8/0.99), `ExponentialLR(0.999)` per epoch, batch 16,
  7040-sample (22-frame) segments, seed 1234;
- data split: train on all of `train-clean-100`, validate on whole utterances
  of all of `dev-clean` (their `wavlm-hifigan-{train,valid}.csv`);
- inference with `k=4`, VAD-trimmed reference set (trigger level 7), output
  loudness -16 LUFS;
- the discriminator, parameter for parameter (70724591): the five period and
  three scale sub-discriminators of the original HiFi-GAN, including spectral norm
  on the first scale discriminator. `KNNVCVocoderModel` rebuilds the period
  discriminators' output convolution with the official `(3, 1)` kernel, which
  `espnet2.gan_tts.hifigan` cannot express (it derives the kernel as
  `kernel_sizes[1] - 1` and requires an odd `kernel_sizes[1]`); pass
  `official_period_output_kernel: false` for the stock `espnet2` `(2, 1)`.

Differences introduced by the ESPnet3 port:

- Each training step runs a single forward pass: the discriminator is updated
  first and then the generator, as in the official script, but the generator's
  adversarial term is computed against the discriminator *before* its update
  (the official script re-runs the updated discriminator).
- The generator's input convolution is initialized to `normal(0, 0.01)` by
  `HiFiGANGenerator.reset_parameters`; the official `Generator` applies
  `init_weights` to the upsamples, residual blocks and output convolution but
  leaves `conv_pre` at PyTorch's default initialization.
- `conf/inference.yaml` defaults to ESPnet3's average of the 3 best
  checkpoints by validation mel loss, which is also what `pack_model`
  publishes; the authors use the final checkpoint instead. Set
  `vocoder_checkpoint: ${exp_dir}/last.ckpt` to match them exactly.
- Evaluation pairs (`kind: conversion`) are 200 seeded random dev-clean
  source/target-speaker pairs with up to 5 minutes of reference audio; the
  paper's exact evaluation lists are not reproduced and no WER/EER metric is
  wired into `measure` yet.
- The discriminator architecture comes from `espnet2.gan_tts.hifigan` and
  matches the official HiFi-GAN design, but the released discriminator weights
  (`do_*.pt`) were not converted, so fine-tuning from them is not supported.

## Scoring

The `measure` stage is generic and ships **no VC metric**, so it needs a
metrics config of your own; running it without one fails with
`Config not provided for stage(s): measure`. The `infer` stage writes
everything a metric needs per test set: `wav.scp` (converted audio),
`ref.scp` (source transcript) and `target_speaker.scp`. A word-error-rate
metric would transcribe `wav.scp` and compare against `ref.scp`; a speaker
metric would compare `wav.scp` against the target speaker's audio.

```yaml
# conf/metrics.yaml
metrics:
  - metric:
      _target_: my_project.metrics.ConvertedSpeechWER
    inputs:
      hyp: wav
      ref: ref
```

## Publishing

```bash
python run.py --stages pack_model \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --publication_config conf/publication.yaml
```

The bundle carries the averaged generator weights, the recipe's `dataset/`
module and the resolved configs, and is loadable from outside the recipe tree:

```python
from espnet3.publication import InferenceModel

model = InferenceModel.from_packed("exp/<tag>/model_pack")
result = model({"speech": source_wav, "reference_speech": [ref_wav, ...]})
converted = result["wav"]
```

Raw `*.ckpt` training states and the prematched feature directory are excluded
from the bundle; regenerate features with `prepare_features`.

## Acknowledgement

All credit for kNN-VC goes to its authors. This recipe reuses:

- the kNN-VC method, training recipe and HiFi-GAN configuration from
  <https://github.com/bshall/knn-vc> (MIT License);
- the WavLM-Large checkpoint (Microsoft, <https://github.com/microsoft/unilm/tree/master/wavlm>)
  and the HiFi-GAN generators trained on (prematched) WavLM features, both
  redistributed by the kNN-VC authors under
  <https://github.com/bshall/knn-vc/releases/tag/v0.1>;
- the WavLM model code, vendored from the kNN-VC repository (which carries it
  over from microsoft/unilm, MIT License) in
  `espnet3/systems/vc/models/knnvc/vendored_wavlm.py`. The model logic is
  unchanged; the two upstream files were merged, their relative import
  resolved, and comments rewrapped to satisfy ESPnet's linters;
- HiFi-GAN (Kong et al., 2020, <https://github.com/jik876/hifi-gan>), used here
  through ESPnet's existing implementation.

Please cite the original paper when using this recipe:

```bibtex
@inproceedings{baas2023knnvc,
  author    = {Matthew Baas and Benjamin van Niekerk and Herman Kamper},
  title     = {Voice Conversion With Just Nearest Neighbors},
  booktitle = {Interspeech},
  year      = {2023},
}
```

Paper: <https://arxiv.org/abs/2305.18975>. Demo page: <https://bshall.github.io/knn-vc/>.
