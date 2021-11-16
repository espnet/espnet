# Usage

Let's say we want to convert from speaker `clb` to `slt`. To get the best performance, we use the Transformer-VC (a.k.a. Voice Transformer Network, VTN) config by training on the complete training set (932 utterances) initialized with a set of pretrained model parameters from M-AILABS judy.

### Training
```
./run.sh --stop_stage 4 --norm_name judy --pretrained_model m_ailabs.judy.vtn_tts_pt --train_config conf/train_pytorch_transformer.tts_pt.single.yaml --tag vtn_tts_pt
```

### Decoding & Evaluation

Let's use the VTN trained after 1000 epochs. We use the pretrained Parallel WaveGAN (PWG) as the vocoder.
```
./run.sh --stage 5 --norm_name judy --train_config conf/train_pytorch_transformer.tts_pt.single.yaml --tag vtn_tts_pt --model snapshot.ep.1000
```

The converted wav files can then be found in `exp/clb_slt_pytorch_vtn_tts_pt/outputs_snapshot.ep.1000_decode_denorm/clb_slt_{dev/eval}/pwg_wav/`.

# FAQ
Q: I ran into the following error. What should I do?
```
...
stage 0: Data preparation
Successfully finished making wav.scp, utt2spk.
Successfully finished making spk2utt.
local/data_prep.sh: line 54: downloads/cmu_us_clb_arctic/etc/arctic.data: No such file or directory
local/data_prep.sh: line 55: downloads/cmu_us_clb_arctic/etc/arctic.data: No such file or directory
...
```

A: This is one strange mistake in the original CMU ARCTIC dataset. `arctic.data` only exists in the `slt` directory. So, you need to manually copy from `downloads/cmu_us_slt_arctic/etc/arctic.data` to `cmu_us_clb_arctic/etc/arctic.data`
