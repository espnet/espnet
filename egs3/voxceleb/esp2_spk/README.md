# VoxCeleb speaker verification recipe

Trains a speaker embedding extractor on the VoxCeleb 1 and 2 development sets
and evaluates it on the cleaned Vox1-O protocol (`veri_test2.txt`).

## Data Preparation

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

### Why the audio is packed

VoxCeleb is ~1.2 M small files. On a shared filesystem, reading them one by
one is bound by per-file metadata I/O, not by bytes: on a Lustre HDD pool we
measured under 9 files/s per reader, which starved 16 dataloader workers and
made a RawNet3 epoch take 4-7.5 h on 2 GPUs. The OmniIO archive holds the same
audio, losslessly as FLAC (0.58x the size of WAV; set `builder.blob.format:
wav` to copy the WAV bytes instead), in a few hundred ~320 MB `blob_<n>.bin`
files, with every utterance's byte range in `metadata.parquet`, so a read is
one seek into a large file, and copying the corpus to node-local disk is a
bulk copy of a few hundred files. Packing is resumable: utterances already in
the archive are skipped. `builder.blob` in `dataset/config.yaml` sets the bin
size and the number of packing workers.

### Training on several corpora

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

`training_rawnet3.yaml` draws mini-batches with
`SpeakerBalancedBatchSampler` (`dataset/sampler.py`), ESPnet2's `catbel`
batching: no speaker appears twice in a batch, and under DDP each rank balances
its own shard. Within that constraint a batch is a uniform sample of the merged
utterance list, so a corpus contributes in proportion to its size.

## Commands

Run the following commands from `egs3/voxceleb/esp2_spk/`.

### RawNet3

RawNet3 trained from scratch on raw waveform through a learnable sinc
filterbank; it downloads no pretrained model.

```bash
# 1) Prepare manifests, trial lists and audio archives (run once)
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

# 5) Pack and upload the model (see conf/publication.yaml)
python run.py --stages pack_model upload_model \
    --training_config conf/tuning/training_rawnet3.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml \
    --publication_config conf/publication.yaml
```

### XEUS and ECAPA-TDNN

ECAPA-TDNN on a jointly fine-tuned XEUS frontend. Run the same commands with
`--training_config conf/tuning/training_xeus_ecapa.yaml`. This config, and only
this one, expects a local XEUS checkpoint:

```bash
hf download espnet/xeus model/xeus_checkpoint.pth --local-dir download
```

Adjust `xeus_checkpoint` in the config if you keep it elsewhere. The encoder is
frozen for the first 5,000 updates and fine-tuned jointly afterwards.

## Notes

- `spk_num` in the training config must equal the number of speakers in the
  label space, that is `wc -l < data/voxceleb12_dev/spk2utt` times the number
  of `speed_perturb_factors`. It is 7,205 for VoxCeleb 1 dev + VoxCeleb 2 dev,
  so 21,615 with the default factors. `create_dataset` logs the number it wrote.
- `training_rawnet3.yaml` speed-perturbs the training set by 0.9/1.0/1.1 on
  the fly, as ESPnet2's `spk.sh` does by default: each perturbed copy of an
  utterance is labelled as a new speaker (`sp0.9-id10001`), which triples both
  the epoch and the label space. Nothing extra is written to disk. Its
  scheduler restarts every 10 epochs at the global batch of 512
  (`num_device x dataloader.train.batch_size`); recompute
  `scheduler.first_cycle_steps` if you change either.
- `valid/eer` is computed each epoch over a strided 10,000-trial subset of
  Vox1-O. Raise `num_trials` for a tighter estimate, at the cost of slower
  epochs. The `infer` stage always scores the full list.
- Trial IDs in `score.scp` are line numbers in
  `data/voxceleb1_test/vox1_o.trials`. `measure` reads each trial's
  target/nontarget label from the same dataset (`dataset:spk_labels`).
- A trained model scores a pair of recordings directly:

  ```python
  from espnet3.publication.inference_model import InferenceModel

  model = InferenceModel.from_packed(
      "exp/training_rawnet3/model_pack", trust_user_code=True
  )
  model({"speech": "enroll.wav", "speech2": "test.wav"})  # {'score': 0.71, ...}
  ```

## Results

EER (%) and minDCF (p_target 0.05) on the cleaned Vox1-O trial list (37,611
trials).

### RawNet3

- Training config: `conf/tuning/training_rawnet3.yaml`
- Inference config: `conf/inference.yaml`
- Metrics config: `conf/metrics.yaml`
- Checkpoint: `last.pth`, the weights after the last (40th) epoch
- Decoding: 4 evenly spaced 8-second crops per utterance, cosine scoring
  averaged over crop pairs, no score normalization

| | EER(%) | minDCF |
|---|---|---|
| vox1_o | 0.755 | 0.0608 |

This reproduces ESPnet2's `egs2/voxceleb/spk1` RawNet3 (0.739% reported). Its
released model, `espnet/voxcelebs12_rawnet3`, scores 0.750% EER under the
protocol above. The result was trained on one GPU (global batch 512, bf16).
`conf/inference.yaml` scores the last epoch rather than
`valid.eer.ave_3best.pth`, as ESPnet2's published result does: the 10,000-trial
validation subset is too noisy to rank late epochs, and that average mixes
checkpoints from different learning-rate cycles (0.888% EER with 3 s x 10
crops).

#### Pretrained model

- [`conf/tuning/training_rawnet3.yaml`](https://huggingface.co/espnet/voxcelebs12_rawnet3_espnet3),
  uploaded with `upload_model.hf_repo: espnet/voxcelebs12_rawnet3_espnet3`
  overriding the default repo name of `conf/publication.yaml`

### XEUS and ECAPA-TDNN

- Training config: `conf/tuning/training_xeus_ecapa.yaml`

Not yet evaluated.
