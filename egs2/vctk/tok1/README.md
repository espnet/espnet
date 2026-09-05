# VCTK Speech Tokenizer

This recipe trains the ASR and speaker-conditioned reconstruction speech
tokenizer on the standard VCTK corpus using the data preparation from
`egs2/vctk/tts1` as the basis for its split policy. The local implementation is
isolated to this recipe and supports the official VCTK 0.92 archive. Every
speaker contributes training utterances and five utterances each to the
development and evaluation sets; only the `mic1` recordings are used.

Legacy `VCTK-Corpus/wav48` data is also detected automatically. That layout
contains one recording per utterance, so no microphone selection is needed.

Run individual stages while validating a new environment:

```bash
./run.sh --stage 1 --stop_stage 1
./run.sh --stage 2 --stop_stage 5
./run.sh --stage 6 --stop_stage 6
./run.sh --stage 7 --stop_stage 7
./run.sh --stage 8 --stop_stage 8
./run.sh --stage 9 --stop_stage 9
./run.sh --stage 10 --stop_stage 10
```

To reuse an external ASR2 K-means model, add
`--centroid_path /path/to/km_2000.mdl`. Stage 7 then validates the model rather
than learning new centroids.
