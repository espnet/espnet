# frontia_vad_v2: voice activity detection on many domains

Voice activity detection trained as single-output diarization (`diar.sh --num_spk 1`),
with the models of [frontia_vad_v1](../../frontia_vad_v1/diar1) (causal TCN, and the
dual-mode TCN whose look-ahead is chosen at run time). v2 trains on 11 domains instead
of meetings only: meetings, conversations, far-field and wearable recordings,
broadcast, read speech, children, human non-speech vocalisations, music, environmental
sound, animals, and quiet close-microphone non-speech. Each epoch draws its 10 s
chunks per domain (`chunk_upsampling_factor: 0.3`), so small domains are upsampled.

## Data preparation (`local/data.sh`)

Every step that makes the data is a stage of `local/data.sh` or a script it calls in
`local/v2/`. The corpora are read straight from their archives (tar, zip, parquet,
RAR) and written into 16 kHz Omni-IO dumps; nothing is unpacked into the corpora
directory.

| stage | what | output |
|---|---|---|
| 0 | download the corpora listed in `local/v2/downloads/<corpus>.tsv`; every file is verified by size and checksum (`local/v2/download.py`) | `${VAD_CORPORA}/<corpus>/` |
| 1 | the meeting corpora of v1 (AMI, ICSI, CHiME-6, NOTSOFAR-1) and the test sets as Kaldi directories (the v1 scripts, `local/data_v1.sh`); the training splits of the meeting corpora as v2 dumps (`local/v2/from_kaldi.py`) | `data/<corpus>_{train,dev,eval}`, `dump/v2/<corpus>/` |
| 2 | augmentation noise (MUSAN noise and music without vocals, CC0 / CC BY / public domain only; all CHiME-3 backgrounds) and room impulse responses (BUT ReverbDB) | `data/noise.scp`, `data/rir.scp`, `data/rir/` |
| 3 | every training corpus from its archives (`local/v2/prep.py`, 8 jobs per corpus) | `dump/v2/<corpus>/<split>/shard<k>/` |
| 4 | the speech of the transcribed corpora by CTC forced alignment, one GPU job per dump (`local/v2/align.py`) | `espnet_rttm`, `align.tsv`, `align_config.json` in each dump |
| 5 | combine all dumps into the training and validation sets; one report per corpus (`local/v2/combine.py`, `local/v2/corpus_report.py`) | `dump/raw/{train,dev}`, `dump/raw/*/domains.tsv`, `dump/v2/reports/<corpus>.json` |

`run.sh` runs `local/data.sh`, then `diar.sh` stage 2 for the test sets only (stage 3
of `diar.sh` rebuilds sets from Kaldi segments and would drop the recordings without
speech), then `diar.sh` from stage 4 (statistics, training, inference, scoring). A
stage skips the corpora it has finished, so it can be rerun, or run for some corpora
at a time: `local/data.sh --stage 3 --stop_stage 3 --corpora npsc,maptask`.

### A dump directory

`local/v2/vad_dump.py` writes, per corpus, split and shard:

- `wav.scp`: Omni-IO entries (16 kHz FLAC); `utt2num_samples`;
- `espnet_rttm`: the speech of each recording as one speaker, in samples, and an `END`
  line for every recording, also recordings without speech (the diar task gives
  those all-zero labels);
- `utt2category`: the domain of each recording, which `chunk_upsampling_factor` balances;
- `manifest.tsv`: one row per recording: corpus, domain, label source, negative type,
  microphone, language, licence, source member, seconds, speech seconds;
- `summary.json`, `skipped.txt` (sources that could not be read or were filtered, with
  the reason); for transcribed corpora `text` and `utt2lang` (before stage 4).

Recordings shorter than 30 s (`--pack_seconds`) are concatenated, in the order they are
read, per language and microphone, into recordings of at least 30 s, with their labels
and transcripts: the training chunks are 10 s, and a shorter recording gives none.

### Labels

Each corpus has one label source:

- **H**: human segments or timestamps (meetings, conversations, broadcast); the speech
  of a recording is the union of all participants' segments, also on close-talk
  channels;
- **D**: known exactly: a clip without speech (non-speech corpora: the whole clip is
  non-speech), or a mixture of known sources;
- **A**: transcripts only (read speech): stage 4 aligns each recording to its
  transcript word by word with OWSM-CTC v4 (`espnet/owsm_ctc_v4_1B` at revision
  `6075bf3d42e6037d1ef7abd7c5558ade886e6eaa`) and CTC segmentation. The speech is the
  union of the word spans, with pauses shorter than 0.2 s counted as speech
  (`--align_min_pause`); a recording with a word scored below -15 is dropped
  (`--align_min_word_score`). On LibriSpeech dev, -15 drops no recording with a
  correct transcript (-10 dropped 9 % of them), while mismatched transcripts score
  -18 or lower. The buffers of many recordings are encoded together
  (`--align_batch_seconds 3840`, fp32), which gives the same alignments as one
  recording at a time (`local/v2/test_align_batch.py`); bf16 changed them.

Human non-speech vocalisations (laughter, coughs, breathing, crying) are non-speech;
singing is speech. Clips of the non-speech corpora that are tagged with speech or
singing are left out.

### Corpora

Training uses only material whose licence allows it without a non-commercial or
share-alike condition (CC0, CC BY, public domain, ODC-BY, CDLA-Permissive, Apache,
MIT, or a data use agreement that allows training); per-clip licences (Freesound)
are filtered to CC0 and CC BY. The one exception is CHiME-6 (CC BY-SA 4.0), whose
share-alike term is planned to be removed.

`local/v2/adapters.py` says which registered corpora are left out (`NOT_USED`, with the
reason), the reader options and the hour caps of the train split: 2,000 h per corpus
(BABYCRY 50 h), 300 h per language in the multilingual corpora, MLS 2,000 h English and
300 h per other language, Libriheavy 700 h per subset. Mandarin uses `<cmn>`; the
official test splits of the corpora are not used for training.

| corpus | directory | domain | label | licence | download | used |
|---|---|---|---|---|---|---|
| adl_ambient | adl_ambient | environment | D | CC BY 4.0 | `downloads/adl_ambient.tsv` | yes |
| aishell3 | aishell3 | read | A | Apache-2.0 | `downloads/aishell3.tsv` | yes |
| arabic_speech_corpus | arabic_speech_corpus | read | A | CC BY 4.0 | - | no: licence unclear (CC BY 4.0 vs BY-NC-SA mirrors) |
| avp | avp | vocal_nonspeech | D | CC BY 4.0 | `downloads/avp.tsv` | yes |
| babycry | babycry | vocal_nonspeech | D | CC BY 4.0 | `downloads/babycry.tsv` | yes |
| bcn_night | bcn_night | environment | H | CC BY 4.0 | `downloads/bcn_night.tsv` | yes |
| bcp | bcp | meeting | D | CC BY 4.0 | `downloads/bcp.tsv` | yes |
| berst | berst | farfield | A | CC BY 4.0 | `downloads/berst.tsv` | yes |
| birdvox | birdvox | animal | D | CC BY 4.0 | `downloads/birdvox.tsv` | yes |
| bsd10k | bsd10k | environment | D | CC BY 4.0 (per-clip CC0/BY) | `downloads/bsd10k.tsv` | yes |
| bsd10k_quiet | bsd10k | quiet_closemic | D | CC BY 4.0 (CC0/BY) | `downloads/bsd10k.tsv` | yes |
| bsd35k_cs | bsd35k_cs | environment | D | CC BY 4.0 (per-clip CC0/BY) | `downloads/bsd35k_cs.tsv` | yes |
| bsd35k_cs_quiet | bsd35k_cs | quiet_closemic | D | CC BY 4.0 (CC0/BY) | `downloads/bsd35k_cs.tsv` | yes |
| cavemove | cavemove | environment | D | CC BY 4.0 | `downloads/cavemove.tsv` | yes |
| children_hri_en | children_hri_en | children | A | CC BY 4.0 | - | no: no consent statement for the children, no transcripts |
| chime3_background | chime3_background | environment | D | CC BY 4.0 | `downloads/chime3_background.tsv` | yes |
| clarity_speech | clarity_speech | read | A | CC BY 4.0 | `downloads/clarity_speech.tsv` | yes |
| clotho | clotho | environment | D | per-clip CC0/BY (Freesound) | - | no: unlabelled background speech; voice filter by keywords only |
| commonlanguage | commonlanguage | read | A | CC BY 4.0 | `downloads/commonlanguage.tsv` | yes |
| commonvoice | commonvoice | read | A | CC0-1.0 | manual (see below) | yes |
| coughvid | coughvid | vocal_nonspeech | D | CC BY 4.0 | `downloads/coughvid.tsv` | yes |
| crowdioset | crowdioset | environment | D | per-clip CC0/BY | `downloads/crowdioset.tsv` | yes |
| datahive_conv | datahive_conv | conversation | H | CC BY 4.0 | - | no: no transcripts or timings |
| dcase2025_bioacoustic | dcase2025_bioacoustic | animal | D | CC BY 4.0 | `downloads/dcase2025_bioacoustic.tsv` | yes |
| dcase_bioacoustic | dcase_bioacoustic | animal | D | CC BY 4.0 | `downloads/dcase_bioacoustic.tsv` | yes |
| designed_vocalizations | designed_vocalizations | vocal_nonspeech | D | CC BY 4.0 (clips CC0 / CC BY 3.0 / CC BY 4.0) | `downloads/designed_vocalizations.tsv` | yes |
| dral | dral | conversation | H | CC0-1.0 | - | no: 2.6 s speech-only fragments, no boundaries to learn |
| droneaudioset | droneaudioset | environment | D | MIT | `downloads/droneaudioset.tsv` | yes |
| egomusic | egomusic | music | D | CC BY 4.0 | - | no: singing without timed vocal labels |
| fleurs | fleurs | read | A | CC BY 4.0 | `downloads/fleurs.tsv` | yes |
| fsd50k | fsd50k | environment | D | CC BY 4.0 (per-clip CC0/BY) | `downloads/fsd50k.tsv` | yes |
| fsd50k_quiet | fsd50k | quiet_closemic | D | CC BY 4.0 (CC0/BY) | `downloads/fsd50k.tsv` | yes |
| fsd50k_vocal | fsd50k | vocal_nonspeech | D | CC BY 4.0 (CC0/BY) | `downloads/fsd50k.tsv` | yes |
| fsdnoisy18k | fsdnoisy18k | environment | D | CC BY (clips CC0/BY) | `downloads/fsdnoisy18k.tsv` | yes |
| fuss | fuss | environment | D | CC BY 4.0 (sources CC0) | `downloads/fuss.tsv` | yes |
| german_kinect | german_kinect | read | A | CC BY 4.0 | `downloads/german_kinect.tsv` | yes |
| harpervalleybank | harpervalleybank | conversation | H | CC BY 4.0 | - | no: segment times come from a machine pipeline |
| holoassist | holoassist | farfield | H | CDLA-Permissive-2.0 | - | no: on hold until the event timing is checked on 3 videos |
| indic_diarbench | indic_diarbench | meeting | H | CC BY 4.0 | `downloads/indic_diarbench.tsv` | yes |
| interaction_behavior | interaction_behavior | vocal_nonspeech | D | labels CC BY 4.0; audio IFADV GPL-2.0 | - | no: IFADV audio is GPL-2.0 (copyleft, treated as SA) |
| kallaama | kallaama | conversation | H | CC BY 4.0 | `downloads/kallaama.tsv` | yes |
| keyboard_sound | keyboard_sound | environment | D | MIT | `downloads/keyboard_sound.tsv` | yes |
| keyboard_sound_quiet | keyboard_sound | quiet_closemic | D | MIT | `downloads/keyboard_sound.tsv` | yes |
| ksc | ksc | read | A | CC BY 4.0 | `downloads/ksc.tsv` | yes |
| libriheavy | librilight | read | A | LibriVox public domain (Libri-Light); Libriheavy text Apache-2.0 | `downloads/librilight.tsv` | yes |
| librispeech | librispeech | read | A | CC BY 4.0 | `downloads/librispeech.tsv` | yes |
| libriwasn | libriwasn | farfield | D | CC BY 4.0 | `downloads/libriwasn.tsv` | yes |
| locata | locata | farfield | D | ODC-BY 1.0 | `downloads/locata.tsv` | yes |
| lsx | lsx | music | D | CC BY 4.0 | `downloads/lsx.tsv` | yes |
| malromur | malromur | read | A | CC BY 4.0 | `downloads/malromur.tsv` | yes |
| maptask | maptask | conversation | H | CC BY 4.0 | `downloads/maptask.tsv` | yes |
| marmosetvox | marmosetvox | animal | D | CC BY 4.0 | `downloads/marmosetvox.tsv` | yes |
| mcvampire | mcvampire | environment | D | CC BY 4.0 | `downloads/mcvampire.tsv` | yes |
| minds14 | minds14 | read | A | CC BY 4.0 | `downloads/minds14.tsv` | yes |
| mls | mls | read | A | CC BY 4.0 | `downloads/mls.tsv` | yes |
| musan | musan | music | D | MUSAN per-file CC0/BY/PD | `downloads/musan.tsv` | yes |
| musan_noise | musan | environment | D | MUSAN per-file CC0/BY/PD | `downloads/musan.tsv` | yes |
| nb_samtale | nb_samtale | meeting | A | CC0-1.0 | `downloads/nb_samtale.tsv` | yes |
| npsc | npsc | broadcast | H | CC0-1.0 | `downloads/npsc.tsv` | yes |
| openmic2018 | openmic2018 | music | D | CC BY 4.0 (per-track CC0/BY) | `downloads/openmic2018.tsv` | yes |
| openyap | openyap | conversation | H | OpenYAP DUA (training OK) | manual (see below) | yes |
| perit_duplex | perit_duplex | conversation | H | CC BY 4.0 | - | no: no transcripts or timings |
| primock57 | primock57 | conversation | H | CC BY 4.0 | `downloads/primock57.tsv` | yes |
| pyramic | pyramic | farfield | D | CC BY 4.0 | - | no: 5 played sentences in an anechoic room, energy labels |
| realised | realised | environment | D | CC BY 4.0 | `downloads/realised.tsv` | yes |
| realman | realman | farfield | A | CC BY 4.0 | `downloads/realman.tsv` | yes |
| ruls | ruls | read | A | Public domain (LibriVox) | `downloads/ruls.tsv` | yes |
| samromur | samromur | read | A | CC BY 4.0 | `downloads/samromur.tsv` | yes |
| samromur_children | samromur_children | children | A | CC BY 4.0 | `downloads/samromur_children.tsv` | yes |
| simsamu | simsamu | conversation | H | MIT | `downloads/simsamu.tsv` | yes |
| slakh2100 | slakh2100 | music | D | CC BY 4.0 | `downloads/slakh2100.tsv` | yes |
| speech_commands | speech_commands | read | A | CC BY 4.0 | `downloads/speech_commands.tsv` | yes |
| speech_commands_background | speech_commands | environment | D | CC BY 4.0 | `downloads/speech_commands.tsv` | yes |
| starss23 | starss23 | farfield | H | MIT | `downloads/starss23.tsv` | yes |
| synth_lownoise | - | quiet_closemic | D | generated (no licence) | generated | yes |
| vocal_imitation | vocal_imitation | vocal_nonspeech | D | CC BY 4.0 | `downloads/vocal_imitation.tsv` | yes |
| vocalsketch | vocalsketch | vocal_nonspeech | D | CC BY 4.0 | `downloads/vocalsketch.tsv` | yes |
| voices | voices | farfield | A | CC BY 4.0 | `downloads/voices.tsv` | yes |
| waxal | waxal | read | A | CC BY 4.0 (University of Ghana parts) | `downloads/waxal.tsv` | yes |
| zeroth_korean | zeroth_korean | read | A | CC BY 4.0 | `downloads/zeroth_korean.tsv` | yes |

Downloads that need an account or an agreement are not in `local/v2/downloads/`:

- **Common Voice** (Scripted Speech 27.0, Spontaneous Speech 5.0; CC0): Mozilla Data
  Collective. Accept the terms of each dataset on its page, then run
  `MDC_API_KEY=... python3 local/v2/download_commonvoice.py ${VAD_CORPORA}/commonvoice/ARCHIVES`
  (7 datasets per day per organisation).
- **OpenYAP**: from its vendor under a data use agreement that allows training.
- **The v1 corpora** (AMI, ICSI, CHiME-6, NOTSOFAR-1 for training; DiPCo, AliMeeting,
  AISHELL-4, VoxConverse, Earnings-21, AVA-Speech, TEN VAD for testing): as in
  frontia_vad_v1; set their paths in `db.sh`. AMI: `../../ami/asr1/local/ami_download.sh`.

### Reproducing the data

- The outputs depend only on the corpora, `local/v2/`, and the options of
  `local/data.sh`; `--nshards` (8) is one of them, since recordings are packed within a
  shard. Dev splits without an official one, gains of the quiet close-mic clips and the
  synthetic noise are seeded by fixed hashes and indices.
- `local/v2/requirements.txt` pins the Python packages the readers need (h5py,
  imageio-ffmpeg, pyarrow, soundfile); 7-Zip (`7zz`, or `SEVENZ=<path>`) reads RAR.
- `dump/v2/reports/<corpus>.json` lists, per corpus, the source files with size and
  checksum, the licence, the hours read and kept per split, and the reasons and
  counts of every exclusion; the reports have no timestamps or paths, so two runs can
  be compared with `diff -r`.

## Running

Set `VAD_CORPORA` (and the v1 corpus paths) in `db.sh`, and the job backend in
`cmd.sh`: `run.pl` runs the jobs on the local machine, `slurm.pl` or `pbs.pl` submit
them to a cluster (`conf/slurm.conf`, `conf/pbs.conf`). Stage 3 runs `--nshards` CPU
jobs per corpus with `${train_cmd}`; stage 4 runs one GPU job per transcribed dump
with `${cuda_cmd} --gpu 1`. On a cluster where a job cannot submit others, run
`local/data.sh --stage N --stop_stage N --corpora <list>` inside a job with
`cmd_backend=local`, so the shards of a stage run in parallel on that node.

```sh
./run.sh                                   # all stages
./run.sh --stage 4                         # training and evaluation only
local/data.sh --stage 3 --stop_stage 3 --corpora npsc,maptask
```

Training (`conf/train.yaml`, `conf/tuning/train_dualmode.yaml`) keeps the 5 best
checkpoints and averages the 2 and 5 best (`keep_nbest_models: [2, 5]`). Evaluation
scores whole recordings (UEM over each recording) on the test sets of frontia_vad_v1;
the headline number is the macro average of DetER over the sets.

Results will follow the training.
