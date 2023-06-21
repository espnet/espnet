- Environments

  - python version: `3.8.12`

  - espnet version: `espnet 0.10.7a1`

  - chainer version: `chainer 6.0.0`

  - pytorch version: `pytorch 1.10.0`



- Model files 

    - training config file (teacher model): `./conf/tuning/finetune_transformer.yaml`

    - training config file: `./conf/tuning/train_conformer_fastspeech2.yaml`

 

- Results: {using our pretrained ASR model on MGB2 data} here is the recipe "egs/mgb2"

  - FastSpeech2 (Trans.) w/ PWG: (R = 1) 

      - CER: 3.9

      - WER: 9.13

      - MOS

        - intelligibility: 4.4 ± 0.06
        - naturalness: 4.2 ± 0.06
