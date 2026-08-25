# VoxCeleb speaker verification recipe

Trains a speaker embedding extractor on the VoxCeleb 1 and 2 development sets
and evaluates it on the cleaned Vox1-O trial list (`veri_test2.txt`).

## Corpus layout

VoxCeleb cannot be downloaded automatically, so point `VOXCELEB` at a root that
holds the three audio trees below, or place them under `download/`:

```
$VOXCELEB/
  voxceleb1/dev/<speaker>/<video>/<utterance>.wav
  voxceleb1/test/<speaker>/<video>/<utterance>.wav
  voxceleb2/dev/<speaker>/<video>/<utterance>.wav
```

Everything must be 16 kHz mono WAV. VoxCeleb2 ships as AAC, so convert it once,
for example with `ffmpeg -i in.m4a -ac 1 -ar 16000 out.wav`.

Noise and reverberation augmentation additionally needs MUSAN and RIRS_NOISES:

```bash
export VOXCELEB=/path/to/voxceleb
export MUSAN=/path/to/musan
export RIRS_NOISES=/path/to/RIRS_NOISES
```

`create_dataset` writes the manifests (`wav.scp`, `utt2spk`, `spk2utt`), the
augmentation lists (`musan_*.scp`, `rirs.scp`), and the converted trial list
under `data/`. It downloads the Vox1-O protocol itself. If MUSAN or
RIRS_NOISES are missing it says so and skips those lists; set
`noise_apply_prob` and `rir_apply_prob` to `0.0` in the training config to
train without them.

## Quick start

```bash
# 1) Prepare manifests and trial lists (run once)
python run.py --stages create_dataset \
    --training_config conf/tuning/training_rawnet3.yaml

# 2) Train
python run.py --stages train \
    --training_config conf/tuning/training_rawnet3.yaml

# 3) Score the Vox1-O trials
python run.py --stages infer \
    --training_config conf/tuning/training_rawnet3.yaml \
    --inference_config conf/inference.yaml

# 4) Compute EER and minDCF
python run.py --stages measure \
    --training_config conf/tuning/training_rawnet3.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

## Configurations

| Config | Frontend | Encoder | Notes |
|---|---|---|---|
| [`conf/tuning/training_rawnet3.yaml`](conf/tuning/training_rawnet3.yaml) | learnable sinc filterbank | RawNet3 | trained from scratch on raw waveform |
| [`conf/tuning/training_xeus_ecapa.yaml`](conf/tuning/training_xeus_ecapa.yaml) | XEUS (jointly fine-tuned) | ECAPA-TDNN | needs the XEUS checkpoint, see below |

The SSL configuration expects a local XEUS checkpoint:

```bash
hf download espnet/xeus model/xeus_checkpoint.pth --local-dir download
```

Adjust `xeus_checkpoint` in the config if you keep it elsewhere. The encoder is
frozen for the first 5,000 updates and fine-tuned jointly afterwards.

## Notes

- `spk_num` in the training config must equal the number of training speakers,
  that is `wc -l < data/voxceleb12_dev/spk2utt`. It is 7,205 for
  `voxceleb12_dev`.
- Checkpoints are selected on `valid/eer`, computed each epoch over a strided
  10,000-trial subset of Vox1-O. Raise `num_trials` for a tighter estimate, at
  the cost of slower epochs. The `infer` stage always scores the full list.
- Trial IDs in `score.scp` and `label.scp` are line numbers in
  `data/voxceleb1_test/vox1_o.trials`.

## Results

| Config | EER(%) | minDCF | Hugging Face |
|---|---|---|---|
| `conf/tuning/training_rawnet3.yaml` | | | |
| `conf/tuning/training_xeus_ecapa.yaml` | | | |
