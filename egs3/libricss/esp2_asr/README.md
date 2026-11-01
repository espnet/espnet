# LibriCSS ASR recipe

Port of [`egs/libri_css/asr1`](https://github.com/espnet/espnet/pull/2246)
(ESPnet1) to ESPnet3. LibriCSS is a **continuous speech separation /
far-field meeting ASR benchmark** built on top of LibriSpeech: ten
~10-minute meetings were simulated at six overlap conditions (`0L`, `0S`,
`OV10`, `OV20`, `OV30`, `OV40` — from non-overlapping to 40% single-speaker
overlap), each captured with a 6-microphone circular array plus one
close-talk channel (7 channels in `record/raw_recording.wav`; ~10 hours of
audio across all 60 recordings).

This recipe is **evaluation-only**: nothing is trained. It decodes with the
ESPnet-model-zoo release of the same pretrained LibriSpeech Transformer
(e18) the ESPnet1 recipe used
(`Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best`,
zenodo record 4030677) and diarizes with a pretrained ESPnet speaker
embedding model, then reports

- **SA-WER** (CHiME-6 style speaker-attributed WER) on the diarized flow, and
- **WER** per overlap condition on the oracle-segment flow,

for `dev` (`session0`) and `eval` (`session1`–`session9`), matching the two
scoring scripts of the original recipe.

## Corpus access

The `create_dataset` stage downloads `for_release.zip` (~6.4 GB, ~17 GB
unpacked) from Google Drive (id `1Piioxd5G_85K9Bhcr8ebdhXx0CnaHy7l`) with
`gdown`, unpacks it, and extracts one microphone channel per meeting
(default: channel 0, set `create_dataset.mic` in `conf/eval.yaml` to
choose another). Unlike egs1, it does **not** run the corpus-shipped
`segment_libricss.py`: that script only writes per-utterance/per-segment wav
copies which this ASR flow never reads.

If you already have the corpus, skip the download by either

```bash
export LIBRICSS=/path/to/libricss   # directory containing for_release/
```

or by setting `create_dataset.source_dir` in `conf/eval.yaml`.

## Requirements

On top of a standard ESPnet install (`tools/`'s `make` installs
`espnet[train]`, which the ESPnet3 runner needs):

```bash
pip install jiwer            # `measure` stage        (or espnet[asr])
pip install gdown            # corpus download        (or espnet[egs2])
pip install webrtcvad-wheels # `segment` stage, mode: vad (provides the
                             # `webrtcvad` module; the original `webrtcvad`
                             # package often fails to build on new Pythons)
pip install scikit-learn     # `diarize` stage, NME spectral clustering
                             #                        (or espnet[egs2])
```

The `diarize` stage needs the ESPnet2 speaker toolkit
(`espnet2.bin.spk_inference.Speech2Embedding`), included in regular ESPnet
installs (`espnet[egs2]` pulls the matching extras). The oracle flow needs
neither `webrtcvad` nor `scikit-learn`.

## Stage flow

```
create_dataset -> segment -> diarize -> infer -> measure
```

`run.py` defines `LibriCSSSystem`, an `ASRSystem` whose declared stages
are exactly this flow: `segment` and `diarize` are recipe-local stages,
and the training stages are not declared. `segment` writes per-recording
JSON manifests to `${exp_dir}/segments/<split>/`; `diarize` consumes them
and writes speaker-labeled manifests to `${exp_dir}/diarized/<split>/`
(plus an `rttm` file usable with external DER tools). Both clear the
manifests of an earlier run first. `dataset/` reads these manifests:
each item is the segment's audio slice (`speech`) plus its metadata
(`utt_id`, `spk`, `reco`, `start`, `end` and, in the oracle flow only,
`text`), declared as the dataset's `fields`. `infer` decodes each segment
into `text.scp` through the shared ESPnet3 inference contract and ASR
runner, both inherited from `egs3/TEMPLATE/esp2_asr/conf/inference.yaml`
(16 kHz mono audio in, text out); the recipe's inference configs only set
the test sets and the pretrained model. `measure` scores the transcripts,
reading the segment metadata from the same dataset as `dataset:<column>`
inputs. `measure` caches those columns under
`<inference_dir>/<split>/dataset/`; the next `infer` run clears them.

### Diarized flow (SA-WER)

webrtcvad segmentation + NME spectral clustering, reproducing the egs1
`diarize.sh --diarizer_type spectral` pipeline:

```bash
python run.py --stages all \
    --eval_config conf/eval.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

Outputs land in `exp/eval/` (`segments/`, `diarized/`, `inference/`),
with per-recording speaker matching details in
`exp/eval/inference/<split>/sawer_details.json`.

### Oracle-segment flow (per-condition WER)

Corpus-provided segments and transcripts; skips `diarize`:

```bash
python run.py --stages create_dataset segment infer measure \
    --eval_config conf/eval_oracle.yaml \
    --inference_config conf/inference_oracle.yaml \
    --metrics_config conf/metrics_oracle.yaml
```

The `*_oracle` configs use `exp_tag: eval_oracle`, so this flow keeps a
separate `exp/eval_oracle/` tree.

### CPU smoke run

For a quick end-to-end validation without full-scale decoding, trim tiny
manifests out of the real `segment`/`diarize` outputs and decode them with
the optional `*_smoke` configs (a few seconds of audio; 1.5–2 min per flow
on CPU on a Ryzen 7 2700X, including the ~25 s model load):

```bash
# after: --stages segment diarize (diarized flow) and
#        --stages segment --eval_config conf/eval_oracle.yaml
python src/make_smoke_manifests.py diarized \
    exp/eval/diarized exp/eval/diarized_smoke 2 4 1.0 8.0
python src/make_smoke_manifests.py oracle \
    exp/eval_oracle/segments exp/eval_oracle/segments_smoke 3 3 1.5 6.0

python run.py --stages infer measure --eval_config conf/eval.yaml \
    --inference_config conf/inference_smoke.yaml \
    --metrics_config conf/metrics.yaml
python run.py --stages infer measure --eval_config conf/eval_oracle.yaml \
    --inference_config conf/inference_oracle_smoke.yaml \
    --metrics_config conf/metrics_oracle.yaml
```

Smoke outputs land in their own `inference_*_smoke/` dirs (config-stem
naming), so canonical results stay untouched. The diarized smoke SA-WER is
meaningless by construction — a handful of turns scored against full
reference streams, with every unmatched reference speaker charged as padded
deletions; the oracle smoke WER is a meaningful, if tiny, sample.

## Configuration

The recipe's root config is `conf/eval.yaml` (oracle variant:
`conf/eval_oracle.yaml`) — named `eval` because nothing is trained here.
`run.py` exposes it as `--eval_config`, an alias of the `--training_config`
flag (the framework slot keeps its name internally); the training-only
template blocks merged in from `egs3/TEMPLATE/esp2_asr`
(optimizer/scheduler/dataloader/trainer/fit) are nulled out in these files
because no declared stage consumes them. Pass `--eval_config` (and, for
`measure`, `--inference_config`) also when running `infer` or `measure` on
their own: the inference and metrics configs take `exp_dir` and
`inference_dir` from them.

Everything recipe-specific lives in the `libricss` block of the eval
configs (see the comments there): VAD aggressiveness/frame/padding
(`libricss.segment.vad.*`), `mode: oracle` to segment from the corpus
transcripts, and the diarizer settings (`libricss.diarize.*`): speaker model
tag, subsegmentation window/period (defaults 1.5 s / 0.75 s / 0.5 s
minimum, matching Kaldi's `extract_xvectors.sh`), and NME-SC parameters
(`num_clusters: null` = automatic cluster count, `max_num_clusters: 10`,
`pmax: 20`).

## Results

Measured with this recipe on the full corpus (beam 20, ASR-only decoding,
`pad_missing_speakers: true`), against the ESPnet1 reference numbers (same
LibriSpeech Transformer ASR + Transformer LM rescoring, Kaldi
`0012_diarization_v1` x-vectors + NME-SC, egs1 scoring):

| flow | metric | split | this recipe | egs1 reference |
| --- | --- | --- | --- | --- |
| oracle | WER | dev | 20.27 | 19.20 |
| oracle | WER | eval | 21.20 | 19.72 |
| diarized (spectral) | SA-WER | dev | 28.88 | 28.07 |
| diarized (spectral) | SA-WER | eval | 29.25 | 27.01 |

The +1–2 point gaps are the expected cost of the deviations below (no LM
rescoring; for the diarized flow, a different speaker-embedding extractor
and a stricter `pad_missing_speakers` default). Both metrics rise steadily
with the overlap ratio — oracle eval WER goes from ~6.9 on the
non-overlapping conditions to 39.03 on `OV40`; SA-WER eval from ~15.0 to
43.17. Scores land in `exp/eval/inference/metrics.json` (`SA_WER` overall,
`SA_WER_<condition>` per overlap condition, e.g. `SA_WER_OV40`) and
`exp/eval_oracle/inference_oracle/metrics.json` (`WER`,
`WER_<condition>`), with per-recording details
in `exp/eval/inference/<split>/sawer_details.json` (diarized) and
`exp/eval_oracle/inference_oracle/<split>/condition_wer_details.json`
(oracle).

The numbers reproduce across hosts and software stacks: a run with torch
2.13.0 and Python 3.12 (Ryzen 5 7500F, RTX 5090) and one with torch 2.14.0
and Python 3.13 (Ryzen 7 2700X, RTX 3090) gave identical oracle WERs and
dev SA-WERs, and eval SA-WERs 0.01 apart (29.24 vs. 29.25), from slightly
different speaker turns on 3 of the 54 eval recordings.

Three documented deviations from egs1:

- **ASR-only decoding**: egs1 rescored the beam with the LibriSpeech
  Transformer LM shipped in its model bundle (`--rnnlm rnnlm.model.best`).
  No espnet2-format release of that LM exists, so this recipe decodes with
  the acoustic model alone.
- `pad_missing_speakers: true` (the `SAWER` default) charges unmatched
  reference speakers as deletions and unmatched hypothesis speakers as
  insertions, following CHiME-6 practice. The egs1 `best_wer_matching.py`
  silently dropped unmatched speakers, biasing SA-WER down; add
  `pad_missing_speakers: false` to the metric in `conf/metrics.yaml` to
  mimic the old behavior.
- The egs1 `make_rttm.py` flat-turn construction (whose index handling can
  drop a segment for two-segment recordings) is reimplemented cleanly in
  `src/diarization.py`; behavior is identical on well-formed inputs.

## Runtime notes

Full corpus (~10 h of audio, 60 recordings), measured on two hosts: a
Ryzen 5 7500F with an RTX 5090, and a Ryzen 7 2700X (12 CPUs) with an RTX
3090. Where they differ, both are given, in that order:

- `create_dataset`: ~2 min (second host) to extract one microphone channel
  of the 60 recordings, once the corpus is downloaded and unpacked.
- `segment`: ~15 / ~20 s for the diarized flow (5,892 webrtcvad segments
  across both splits), ~3 s / under 1 s for the oracle flow.
- `diarize`: ~1.2 / ~2.4 min for all 60 recordings (~41 k x-vector
  embeddings of 1.5 s subsegments, one call each — `Speech2Embedding` has
  no batched interface; `device: auto` uses CUDA when available). The NME
  sweep runs ~19 small LAPACK eigendecompositions per recording, which
  OpenBLAS makes drastically slower under thread oversubscription
  (`eigvalsh` on a 750×750 matrix: ~16 ms at 1–2 threads vs ~5.8 s at 12),
  so `src/spectral_clustering.py` caps BLAS/OpenMP threads at 2 via
  `threadpoolctl` (ships with scikit-learn; no-op if absent) and uses
  vectorized neighbor/threshold helpers. Outputs are byte-identical to the
  uncapped code (verified on the full corpus); on the first host, the cap
  cuts this stage from ~34 min to ~1.2 min.
- `infer`: ~27 / ~48 min for the diarized flow (8,715 turns, 9.3 h) and
  ~45 / ~68 min for the oracle flow (5,023 utterances, 10.4 h) on GPU —
  14–21× / 9–12× real time at `batch_size: 4`, whose peak GPU memory is
  ~5.4 GiB (a batch of the four longest oracle segments, ~34 s each);
  `batch_size: 1` needs ~3.3 GiB. The beam search is bound by single-thread
  CPU speed (the GPU stays at ~30% utilization while decoding takes one
  core), which is where most of the difference between the hosts comes
  from. On CPU, decoding runs at ~0.5× real time per stream (beam 20), so
  use a GPU for the full corpus.
- `measure`: ~10–20 s the first time, which reads the `dataset:` columns
  from the test set; ~1 s afterwards, from the cached columns.

As a diarization sanity signal, NME-SC estimated exactly 8 speakers — the
true per-meeting count — on 5 of the 6 dev recordings (10 on `OV40`, the
hardest condition) and on the large majority of the 54 eval recordings.

## Credits

Original ESPnet1 recipe: [espnet#2246](https://github.com/espnet/espnet/pull/2246).
Ported components retain their upstream provenance: `spec_clust.py` and
`calc_cossim_scores.py` (Maxim Korenevsky, STC-innovations Ltd),
`make_rttm.py` (David Snyder, Matthew Maciejewski),
`get_perspeaker_output.py` / `best_wer_matching.py` (Desh Raj),
`multispeaker_score.sh` / `score_reco_diarized.sh` (Ashish Aranda, Yusuke
Fujita, Desh Raj), all Apache 2.0. LibriCSS corpus: Chen et al., "A
continuous speech separation (CSS) approach to the LibriMix corpus,"
arXiv:2010.14445.
