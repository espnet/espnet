### Additional Dependencies

1. pip install textgrid
2. pip install pyannote.metrics (for scoring)

### Download AMI

You can easily download AMI using lhotse with:

```
lhotse download ami --mic ihm-mix,mdm
```

### Download LibriSpeech

In this recipe we use LibriSpeech for multi-speaker simulation to increase the training material.
We also make use of LibriSpeech word alignments from Montreal Forced Alignment (MFA) available at https://zenodo.org/records/2619474

1. LibriSpeech:

```
lhotse download librispeech --full <YOUR_LIBRISPEECH_DIR>
```

2.  LibriSpeech MFA Alignments:

```
wget https://zenodo.org/records/2619474/files/librispeech_alignments.zip?download=1 -O <YOUR_ALIGN_DIR>/librispeech_alignments.zip
unzip <YOUR_ALIGN_DIR>/librispeech_alignments.zip -d <YOUR_ALIGN_DIR>/LibriSpeech_MFA
```

### Training


### Inference

pass 







