# VoxtLM RECIPE

This is the recipe of [VoxtLM](https://arxiv.org/pdf/2309.07937.pdf), unified decoder-only models for consolidating speech recognition, synthesis and speech, text continuation tasks.

   <summary>bib info</summary>

   ```
    @article{maiti2023voxtlm,
  title={Voxtlm: unified decoder-only models for consolidating speech recognition/synthesis and speech/text continuation tasks},
  author={Maiti, Soumi and Peng, Yifan and Choi, Shukjae and Jung, Jee-weon and Chang, Xuankai and Watanabe, Shinji},
  journal={arXiv preprint arXiv:2309.07937},
  year={2023}
}
   ```
   </details>


## Pretrained models
### VoxtLM_OPT350_Kmeans_200
Model link: [VoxtLM_OPT350_Kmeans_200](https://huggingface.co/soumi-maiti/voxtlm_k200/tree/main/exp/opt_350)

### VoxtLM_OPT1.3b_Kmeans_200
Model link: [VoxtLM_OPT1.3b_Kmeans_200](https://huggingface.co/soumi-maiti/voxtlm_k200/tree/main/exp/opt_1.3b)

### VoxtLM_OPT350_Kmeans_1000
Model link: [VoxtLM_OPT350_Kmeans_1000](https://huggingface.co/soumi-maiti/voxtlm-k1000)



See the following pages for the usage:



## Recipe flow

VoxtLM recipe consists of 10 stages.

### 1. Data preparation

Data preparation stage.

If you want to add your own dataset, please create a new folder corresponding to the dataset `$your_own_dataset$` with data process bash scripts in `local`. And then add `data_${your_own_dataset}.sh` in `local`.

**Download vs. reusing an existing local copy**: each `local/data_${dataset}.sh` script checks for a marker before downloading anything (e.g. `${LIBRISPEECH}/LibriSpeech/LICENSE.TXT`, `${LIBRITTS}/LibriTTS/.complete`, `${VCTK}/VCTK-Corpus/wav48_silence_trimmed`, `${LIBRILIGHT}/.complete`). If the marker is already there, the download is skipped entirely and the script goes straight to data prep. So if you already have a copy of a corpus somewhere on disk (e.g. a shared cluster copy), point the corresponding `db.sh` variable at a directory named `downloads` (the default used throughout this recipe) and make that a symlink to the existing copy instead of letting the script fetch it again, e.g.:

```bash
mkdir -p downloads
ln -s /path/to/existing/LibriSpeech       downloads/LibriSpeech
ln -s /path/to/existing/LibriTTS          downloads/LibriTTS
ln -s /path/to/existing/VCTK-Corpus       downloads/VCTK-Corpus   # must contain wav48_silence_trimmed/ (v0.92 layout)
ln -s /path/to/existing/LibriLight/small  downloads/small
ln -s /path/to/existing/LibriLight/medium downloads/medium
ln -s /path/to/existing/LibriLight/large  downloads/large
```

If the marker is missing, `local/data_vctk.sh` will download the official CSTR VCTK Corpus v0.92 (Edinburgh DataShare, `wav48_silence_trimmed` layout — not the older `wav48/` layout that upstream ESPnet's `data_download.sh` fetches, which doesn't match this recipe's `data_prep_0.92.sh`) directly, verify the archive with `unzip -tq`, and extract it under `${VCTK}/VCTK-Corpus/`.


### 2. Wav dump / Embedding preparation

Wav dumping stage.
This stage reformats `wav.scp` in data directories.

### 3. Perform kmeans and get discrete tokens

You can change the kmeans cluster numbers via `--nclusters`.

### 4. Prepare data for different training tasks

Format data for different training tasks for train, valid, and test sets.
Preprare bpe training data.

### 5. BPE training stage

Train BPE using BPE training set obtained from last stage.

### 6. Data statistics collection

Statistics calculation stage.

### 7. Training stage

TTS model training stage.
You can change the training setting via `--lm_config` option.

### 8. Decoding for textlm and speechlm tasks.

Decoding stage.
8.a decodes for textlm task and calculates perplexity for textlm.
8.b decodes for speechlm task and calculates perplexity for speechlm.

### 9. Decoding for ASR task.

Decoding stage for ASR.
You may change the decoding setting via `--lm_inference_asr_config`. The results will be stored in the `${_scoredir}/result.txt`

### 9. Decoding for TTS task.

Decoding stage for TTS.
You may change the decoding setting via `--lm_inference_tts_config`. You may need an extra discrete vocoder to generate wavform from discrete tokens.


### 10-12. (Optional) Pack results for upload

Packing stage.
It packs the trained model files and uploads to [Zenodo](https://zenodo.org/) (Zenodo upload will be deprecated).
If you want to run this stage, you need to register your account in zenodo.

## Concrete run commands & setup (JHU CLSP cluster, D_Bal / OPT-350M reproduction)

This section lists the command for each stage and what needs to be set up before it will work. `--lm_config conf/tuning/train_transformer_opt350.yaml` reproduces the paper's OPT-350M-initialized main results; drop it (or point it at `conf/train_transformer_size768_e12.yaml`) for the from-scratch ablation instead.

Note that `local/data.sh` (invoked by `lm.sh`'s Stage 1, and overridable via `run.sh --local_data_opts`) has its own internal stage numbering, separate from `run.sh`'s: its stage 1 downloads/prepares each of the 4 corpora individually (`local/data_librispeech.sh` etc.), and its stage 2 combines them into `data/train`/`dev`/`test` (this is also where `--data_config` subsampling happens). `./run.sh --stage 1 --stop_stage 1` runs both by default. If a given corpus is already prepared and you only want to re-run the combine step (e.g. after adding a new corpus, or to redo `--data_config` subsampling), skip straight to it with `--local_data_opts "--stage 2"` — don't use that shortcut on a fresh checkout, since it never runs the per-corpus download/prep at all.

**A note on SLURM config files**: `cmd_backend` in `cmd.sh` defaults to `local` (no job scheduler), and this recipe does not ship a `conf/slurm.conf` or any of the other SLURM configs referenced below — each site's partition names, account names, and memory limits are different, so you write your own for your cluster (set `cmd_backend='slurm'` in `cmd.sh` once you have one). The commands below assume you've created:
- `conf/slurm.conf` — a normal GPU/CPU config (see `egs2/TEMPLATE/asr1/conf/slurm.conf` in this repo for a starting template with the `option gpu=*`/`option gpu=0` structure `slurm.pl` expects).
- `conf/slurm_kmeans.conf` — routes to a partition/node with **~120GB+ RAM** (`learn_kmeans.py` loads the whole training pool into memory to fit); no GPU is actually needed for this, just the RAM.
- `conf/slurm_clsp_a100.conf` — routes to a GPU with **~80GB VRAM** for Stage 7 training at this model size/batch config, and must include an `option name=* --job-name $0` line (`espnet2.bin.launch` passes `--name`, and `slurm.pl` errors out without a mapping for it).

| Stage | Command | Setup required first |
|---|---|---|
| 1. Data prep | `./run.sh --stage 1 --stop_stage 1` | `db.sh` corpus paths set up (symlinks or fresh download, see above). To reproduce the paper's D_Bal/D_3M/D_Set scale instead of the full corpus, additionally run `./local/data.sh --stage 2 --stop_stage 2 --data_config bal` (or `3m`/`set`) afterward — this re-subsamples `speechlm`/`textlm` train down to the paper's sizes without touching ASR/TTS data. |
| 2. Wav dump | `./run.sh --stage 2 --stop_stage 2` | `resampy` and `h5py` installed in the espnet conda env (TTS resampling / kmeans feature dumping import them and aren't in the base install). |
| 3. Kmeans + labeling | `./run.sh --stage 3 --stop_stage 3` — **on a memory-constrained cluster this will get through feature dumping and then OOM at the k-means fit itself** (`learn_kmeans.py` needs the whole pool in RAM). When it does, finish just that one step with more memory, then resume: `scripts/feats/perform_kmeans.sh --stage 2 --stop-stage 2 --train_set train --dev_set dev --other_sets test --datadir dump/audio_raw/kmeans_pool --featdir dump/extracted/kmeans_pool --audio_format flac --feature_type hubert_base --layer 6 --feature_conf '{type=s3prl,conf={s3prl_conf={upstream=hubert_base},download_dir=ckpt,multilayer_feature=False,layer=6}}' --km_dir exp/kmeans/hubert_base_6_1000clusters --portion 1.0 --nclusters 1000 --storage_save_mode true --use_gpu true --nj 16 --cpu_cmd "slurm.pl --config conf/slurm_kmeans.conf" --cuda_cmd slurm.pl`, then `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ./run.sh --stage 3 --stop_stage 3 --learn_kmeans false` to skip rebuilding the pool/retraining k-means and just label asr/tts/speechlm with the model that now exists. (If your `conf/slurm.conf`'s default GPU/CPU partitions already have enough RAM, the plain one-liner is all you need — no OOM, no recovery step.) | See the SLURM config note above for `conf/slurm_kmeans.conf`. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps avoid GPU-memory-fragmentation OOM during labeling on a shared/contended GPU. |
| 4. Assemble multitask text | `./run.sh --stage 4 --stop_stage 4` | none |
| 5. BPE training | `./run.sh --stage 5 --stop_stage 5` | none |
| 6. Collect stats | `./run.sh --stage 6 --stop_stage 6` | none — but see the Stage 7 note below: the resulting `exp/lm_stats_*/train/text_shape.bpe` needs a length filter before training, or self-attention memory blows up. |
| 7. LM training | `export HF_HOME=$PWD/hf_cache; export HF_HUB_OFFLINE=1; ./run.sh --stage 7 --stop_stage 7 --ngpu 1 --lm_config conf/tuning/train_transformer_opt350.yaml` | 1) **cmd.sh**: `cuda_cmd` pointed at `conf/slurm_clsp_a100.conf` (see SLURM config note above). 2) **OPT weights**: since GPU compute nodes commonly have no internet, pre-download `facebook/opt-350m` from a login node first: `mkdir -p hf_cache && HF_HOME=$PWD/hf_cache python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/opt-350m')"` — then training with `HF_HUB_OFFLINE=1` (as in the command) never tries the network. 3) **Sequence-length filter**: self-attention memory is O(L²), so `batch_type: length`'s `batch_bins` (a simple linear token-count budget) badly underestimates memory for long sequences. After Stage 6, filter the long tail out before training: `F=exp/lm_stats_en_bpe10000/train/text_shape.bpe; [ -e "$F.orig" ] || cp "$F" "$F.orig"; awk '{split($2,a,","); if (a[1]<=1500) print}' "$F.orig" > "$F"` (safe to rerun: only backs up `$F` to `$F.orig` the first time, and always re-filters from that untouched original — a second run won't clobber the backup with an already-filtered file) (drops ~0.4% of utterances; adjust the path if `--lang`/`--nbpe` differ from the defaults). Without this it OOMs on a full A100 even with `batch_bins` as low as 10000. `dev`/`valid` shapes don't need this (naturally short). |
| 8. Perplexity (textLM/speechLM) | `./run.sh --stage 8 --stop_stage 8 --ngpu 1 --lm_config conf/tuning/train_transformer_opt350.yaml --inference_lm valid.acc.best.pth` | Explicit `--inference_lm valid.acc.best.pth`: the default `valid.acc.ave.pth` is only generated once training reaches `max_epoch` or an (unconfigured) early-stopping trigger, so unless you let training run to completion it won't exist. This stage is a single forward pass (`batch_size=1`, fp32) — it does **not** need an A100; see the GPU-partition note below if you want to run it on a smaller/regular GPU. |
| 9-10. ASR/TTS decoding | `./run.sh --stage 9 --stop_stage 10 --ngpu 1 --lm_config conf/tuning/train_transformer_opt350.yaml --inference_lm valid.acc.best.pth` | Same `--inference_lm` note as Stage 8. `sclite` (via `path.sh`) is used for ASR WER scoring. Stage 10 (TTS) only produces discrete-unit token sequences (`hyp.tok`) — turning those into audio for CER/MOSNet needs a matching HiFiGAN unit-vocoder, which this recipe does not include. |

**Running Stage 8-10 on a non-A100 GPU** (e.g. if your A100 allocation/quota is exhausted): these stages are light enough for any GPU (verified on an 11GB GTX 1080 Ti). Since `lm.sh` re-sources `cmd.sh` internally, you can't override `cuda_cmd` with a plain `export`; instead temporarily point it at the regular `gpu` partition and back, e.g.:

```bash
trap 'sed -i "s#cuda_cmd=\"slurm.pl --config conf/slurm.conf\"#cuda_cmd=\"slurm.pl --config conf/slurm_clsp_a100.conf\"#" cmd.sh' EXIT
sed -i 's#cuda_cmd="slurm.pl --config conf/slurm_clsp_a100.conf"#cuda_cmd="slurm.pl --config conf/slurm.conf"#' cmd.sh
./run.sh --stage 8 --stop_stage 8 --ngpu 1 --lm_config conf/tuning/train_transformer_opt350.yaml --inference_lm valid.acc.best.pth
```

the `trap` guarantees `cmd.sh` is switched back to the A100 config on exit (success, failure, or interrupt) so a later Stage 7 resume doesn't silently run on too little GPU memory.

**Reporting ASR WER split by test-clean/test-other** (the paper reports these separately, but Stage 9's `sclite` run scores the combined `test` set): re-score the existing `hyp.trn`/`ref.trn` from Stage 9 without re-decoding, using the utterance-ID lists from the pre-`combine_data.sh` intermediates:

```bash
SCOREDIR=exp/lm_train_transformer_opt350_en_bpe10000/decode_test_asr/decode_lm_asr/score_wer
mkdir -p "${SCOREDIR}/clean" "${SCOREDIR}/other"
awk '{print "(asr_"$1")"}' data/librispeech/asr/test_clean/text | sort > "${SCOREDIR}/test_clean_ids.txt"
awk '{print "(asr_"$1")"}' data/librispeech/asr/test_other/text | sort > "${SCOREDIR}/test_other_ids.txt"
for split in clean other; do
    idfile="${SCOREDIR}/test_${split}_ids.txt"
    grep -F -f "${idfile}" "${SCOREDIR}/hyp.trn" > "${SCOREDIR}/${split}/hyp.trn"
    grep -F -f "${idfile}" "${SCOREDIR}/ref.trn" > "${SCOREDIR}/${split}/ref.trn"
    sclite -r "${SCOREDIR}/${split}/ref.trn" trn -h "${SCOREDIR}/${split}/hyp.trn" trn \
        -i rm -o all stdout > "${SCOREDIR}/${split}/result.txt"
done
```

(adjust `SCOREDIR`'s `lm_train_..._en_bpe10000` component if you used a different `--lm_config`/`--nbpe`). This is pure local post-processing — no GPU, no `sbatch` needed.

### Known deviations from the paper's full eval suite

- **sWUGGY / sBLIMP** (lexical/syntactic zero-shot probing): not implemented anywhere in ESPnet. The official data is the `sLM21` benchmark package (`https://download.zerospeech.com/datasets/sLM21.dataset.zip`, ~30.6GB, no registration needed) plus the `zerospeech-benchmarks` (`zrc`) scoring toolkit; both `lexical/` (sWUGGY) and `syntactic/` (sBLIMP) items ship with paired text (word/sentence) alongside audio, so both the textLM and speechLM tracks can be evaluated from this one download. Not yet wired up to this recipe.
- **TTS CER / MOSNet**: require converting Stage 10's discrete-unit output into actual audio via a HiFiGAN vocoder (+ x-vector speaker embedding, per the paper), then an external ASR (CER) and MOSNet (quality). None of this is implemented here yet — Stage 10 only gets as far as the unit sequences.
