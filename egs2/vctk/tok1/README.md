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

The layout is detected automatically. Use `--microphone mic2` to select the
second microphone in VCTK 0.92.

To reuse existing k-means centroids, add
`--centroid_path /path/to/km_2000.mdl`.

See the [tok1 template](../../TEMPLATE/tok1/README.md) for pipeline stages and
options.
