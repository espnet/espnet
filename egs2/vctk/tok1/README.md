# VCTK Phonological Tokenizer

This recipe trains a phonological tokenizer on VCTK with ASR and
speaker-conditioned waveform reconstruction losses.

Set `VCTK` in `db.sh` to the corpus or download directory, then run:

```bash
./run.sh --tok_config conf/train_gan_default.yaml
```

## Data preparation

The recipe supports both common VCTK layouts:

- VCTK 0.92: `wav48_silence_trimmed/*/*_mic1.flac`
- Legacy VCTK: `wav48/*/*.wav`

The layout is detected automatically. For VCTK 0.92, this recipe uses the
`mic1` recordings.

To reuse existing k-means centroids, set `quantizer_conf.centroid_path` in the
training config:

```yaml
quantizer_conf:
    centroid_path: /path/to/km_2000.mdl
```

Then set `--use_default_centroid false` to skip the default k-means training.

See the [tok1 template](../../TEMPLATE/tok1/README.md) for pipeline stages and
options.
