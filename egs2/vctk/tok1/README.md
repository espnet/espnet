# VCTK Phonological Tokenizer

This recipe trains a phonological tokenizer on VCTK with ASR and
speaker-conditioned waveform reconstruction objectives.

Set `VCTK` in `db.sh` to the corpus or download directory, then run:

```bash
./run.sh --tok_config conf/train_gan_default.yaml
```

To reuse existing k-means centroids, add
`--centroid_path /path/to/km_2000.mdl`.

See the [tok1 template](../../TEMPLATE/tok1/README.md) for pipeline stages and
options.
