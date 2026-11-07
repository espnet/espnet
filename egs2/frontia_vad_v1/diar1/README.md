# Voice activity detection as single-output diarization

Voice activity detection trained with `diar.sh --num_spk 1`: the model has one
sigmoid output (speech / non-speech) per 8 ms frame, and the label of a frame is
speech when anyone in the recording speaks. Two models:

- **Causal** (`conf/train.yaml`): every block is causal (no STFT centring, global
  mean/variance normalisation, a causal TCN encoder, labels aggregated over the
  same frames). The look-ahead is one STFT window (32 ms); about 1 s of past
  context. 184k parameters.
- **Dual-mode** (`conf/tuning/train_dualmode.yaml`): the same encoder with a
  look-ahead chosen at run time, one set of weights. Online: look-ahead 0
  (`conf/decode.yaml`); offline: 1 s (`conf/tuning/decode_offline.yaml`).

Training reads 10 s windows straight from the Omni-IO archives that stage 2 dumps
(`local/train_windowed.sh`, run.sh stage 5); inference runs on whole recordings.
Scoring is md-eval with no collar. `local/show_vad_result.py` picks one threshold
and median filter on AMI Mix-Headset dev and reports every set at that setting,
next to each set's own best (`RESULTS_vad.md` in the experiment directory).

## Data

Set the corpus paths in `db.sh`. `local/data.sh --corpora <list>` selects the
corpora; run.sh passes all of them. Labels are the union of all speakers'
segments of a recording, written as one pseudo speaker.

Training (and validation: `data/train`, `data/dev`):

| corpus | `db.sh` | licence | microphones | splits |
|---|---|---|---|---|
| AMI | `AMI` | CC BY 4.0 | every wav: Headset-*, Lapel-*, Array1-0*, Array2-0*, Mix-Headset | full-corpus-ASR partition (`../../ami/asr1/local/split_*.orig`) |
| CHiME-6 (OpenSLR SLR150) | `CHIME6` | CC BY-SA 4.0 | every channel of the six arrays and of the binaural in-ear microphones | train / dev / eval |
| NOTSOFAR-1 (240825.1) | `NOTSOFAR1` | CC BY 4.0 | close-talk, every channel of the 7-channel arrays, single-channel devices | train / dev / eval |
| ICSI | `ICSI` | CC BY 4.0 | every chan*.sph and Mix-Headset | lhotse partition (`local/icsi_split_*.txt`) |

Noise for on-the-fly augmentation (`data/noise.scp`, SNR -5 to 20 dB):

| corpus | `db.sh` | licence | part used |
|---|---|---|---|
| MUSAN | `MUSAN` | CC / public domain (per-file LICENSE) | `noise/`, `music/` (`speech/` is left out: babble added to a non-speech region would be labelled non-speech) |
| CHiME-3 background noise | `CHIME3_BACKGROUND` | CC BY 4.0 | all channels |

Evaluation only (test sets; the dev/eval splits of the training corpora are test
sets too). Multi-channel recordings are scored on their first channel.

| corpus | `db.sh` | licence | sets | reference |
|---|---|---|---|---|
| DiPCo | `DIPCO` | CDLA-Permissive-1.0 | dev, eval | every array channel and close-talk |
| AMI Mix-Headset, BUT setup | `AMI`, `AMI_DIAR_SETUP` | AMI: CC BY 4.0; setup: Apache-2.0 | dev, eval (BUT dev, test) | `only_words` RTTMs, as in Bredin and Laurent, Interspeech 2021 |
| AliMeeting (OpenSLR SLR119) | `ALIMEETING` | CC BY-SA 4.0 | dev (Eval_Ali), eval (Test_Ali) | far-field TextGrids |
| AISHELL-4 (OpenSLR SLR111) | `AISHELL4` | CC BY-SA 4.0 | eval (test) | RTTMs of the release |
| VoxConverse | `VOXCONVERSE` | CC BY 4.0 | dev, eval (test) | RTTMs of `joonson/voxconverse` |
| Earnings-21 | `EARNINGS21` | CC BY-SA 4.0 | eval | RTTMs of `revdotcom/speech-datasets` |
| AVA-Speech | `AVA_SPEECH` | labels CC BY 4.0; audio of YouTube movies | eval (15:00-30:00 of each movie) | every label except NO_SPEECH is speech |
| TEN VAD testset | `TEN_VAD` | Apache-2.0 with the additional conditions of the ten-vad repository | eval | frame labels (`.scv`) |

Archives that unpack into many small files (AliMeeting, AISHELL-4) can stay
packed: `local/data.sh` extracts only the members it needs into `downloads/`.

## Running

```bash
./run.sh                                        # causal model, stages 1-7
./run.sh --train_config conf/tuning/train_dualmode.yaml --diar_tag train_dualmode
```

Offline scores of the dual-mode model: see the comment at the top of `run.sh`.
Inference of a set runs in `--inference_nj` jobs. With `--gpu_inference true`,
`run.pl` runs one job per GPU; `--inference_max_jobs_run 8` runs 8 at once (one
2.5 h recording needs about 2 GB of GPU memory with this model).

The dual-mode model needs the TCN encoder with a run-time look-ahead and the
batch augmentation in the model (`wav_augment`).
