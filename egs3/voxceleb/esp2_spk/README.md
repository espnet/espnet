# VoxCeleb speaker verification recipe

Trains a speaker embedding extractor on the VoxCeleb 1 and 2 development sets
and evaluates it on the cleaned Vox1-O protocol (`veri_test2.txt`).

## Corpus layout

VoxCeleb cannot be downloaded automatically, so point `VOXCELEB` at a root that
holds the three audio trees below, or place them under `download/`:

```
$VOXCELEB/
  voxceleb1/dev/<speaker>/<video>/<utterance>.wav
  voxceleb1/test/<speaker>/<video>/<utterance>.wav
  voxceleb2/dev/<speaker>/<video>/<utterance>.m4a
```

VoxCeleb1 is already 16 kHz mono WAV. VoxCeleb2 ships as AAC, which soundfile
cannot read, so `create_dataset` decodes it with ffmpeg into
`data/converted/voxceleb2_dev/` and points the manifests there; the corpus
itself is never written to. That needs `ffmpeg` on `PATH`, and it is the slow
part of the stage, so it runs through `espnet3.parallel`:

```yaml
# In the training config, to widen or narrow the local cluster:
create_dataset:
  n_workers: 32
```

Files that are already converted are skipped, so an interrupted run resumes.
Set `builder.parallel.env` in `dataset/config.yaml` to a job-scheduler backend
(`slurm`, `sge`, ...) to decode on a cluster instead of one machine.

Noise and reverberation augmentation additionally needs MUSAN and RIRS_NOISES:

```bash
export VOXCELEB=/path/to/voxceleb
export MUSAN=/path/to/musan
export RIRS_NOISES=/path/to/RIRS_NOISES
```

`create_dataset` writes the manifests (`wav.scp`, `utt2spk`, `spk2utt`), the
augmentation lists (`musan_*.scp`, `rirs.scp`), and the converted trial list
under `data/`, then packs each split's audio into an
[OmniIO](https://github.com/wavlab-speech/omniio) archive under
`data/<split>/blob/`, which is what training and scoring read. It downloads the Vox1-O protocol itself. If MUSAN or
RIRS_NOISES are missing it says so and skips those lists; set
`noise_apply_prob` and `rir_apply_prob` to `0.0` in the training config to
train without them.

## Why the audio is packed

VoxCeleb is ~1.2 M small files. On a shared filesystem, reading them one by
one is bound by per-file metadata I/O, not by bytes: on a Lustre HDD pool we
measured under 9 files/s per reader, which starved 16 dataloader workers and
made a RawNet3 epoch take 4-7.5 h on 2 GPUs. The OmniIO archive holds the same
WAV bytes (copied, not re-encoded) in a few hundred ~320 MB `blob_<n>.bin`
files, with every utterance's byte range in `metadata.parquet`, so a read is
one seek into a large file, and copying the corpus to node-local disk is a
bulk copy of a few hundred files. Packing is resumable: utterances already in
the archive are skipped. `builder.blob` in `dataset/config.yaml` sets the bin
size and the number of packing workers.

## Training on several corpora

Splits are never concatenated on disk. List them under `dataset.train` and
ESPnet3 merges them with `CombinedDataset`, which is how the default configs
train on both VoxCeleb development sets:

```yaml
dataset:
  train:
    - data_src_args:
        split: voxceleb1_dev
    - data_src_args:
        split: voxceleb2_dev
```

The one thing a merge cannot infer is the speaker label space: `SpkPreprocessor`
maps speaker names to class indices through a single `spk2utt`, and `spk_num`
is its line count. So `dataset/config.yaml` declares the union under
`builder.speaker_unions`, and `create_dataset` writes
`data/voxceleb12_dev/spk2utt` holding nothing but that label space. Add a
corpus by adding it to `builder.sources`, to the union, and to `dataset.train`.

Mini-batches over a merged training set are drawn by the plain shuffling
sampler of `dataloader.train`, so a batch is speaker-balanced only in the
statistical sense: it is a uniform sample of the merged utterance list, and a
corpus contributes in proportion to its size. That is what the reference
RawNet3 result below was trained with. If you need an explicit guarantee of
how many speakers appear per batch, or per corpus, that is what ESPnet3's
`CategoryIterator` is for; wire it up under `dataloader.train.iter_factory`.

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

`training_xeus_ecapa.yaml`, and only that config, expects a local XEUS
checkpoint. The default RawNet3 config consumes raw waveform through a
learnable sinc filterbank and downloads no pretrained model at all:

```bash
hf download espnet/xeus model/xeus_checkpoint.pth --local-dir download
```

Adjust `xeus_checkpoint` in the config if you keep it elsewhere. The encoder is
frozen for the first 5,000 updates and fine-tuned jointly afterwards.

## Notes

- `spk_num` in the training config must equal the number of speakers in the
  label space, that is `wc -l < data/voxceleb12_dev/spk2utt`. It is 7,205 for
  VoxCeleb 1 dev + VoxCeleb 2 dev. `create_dataset` logs the number it wrote.
- Checkpoints are selected on `valid/eer`, computed each epoch over a strided
  10,000-trial subset of Vox1-O. Raise `num_trials` for a tighter estimate, at
  the cost of slower epochs. The `infer` stage always scores the full list.
- Trial IDs in `score.scp` are line numbers in
  `data/voxceleb1_test/vox1_o.trials`. `measure` reads each trial's
  target/nontarget label from the same dataset (`dataset:spk_labels`).
- A trained model scores a pair of recordings directly:

  ```python
  from espnet3.systems.esp2_spk.inference import Inference

  model = Inference.from_pretrained("exp/training_rawnet3/model_pack")
  model("enroll.wav", "test.wav")  # {'score': 0.71}
  ```

## Results

| Config | EER(%) | minDCF | Hugging Face |
|---|---|---|---|
| `conf/tuning/training_rawnet3.yaml` | | | |
| `conf/tuning/training_xeus_ecapa.yaml` | | | |
