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

This section lists the actual command used for each stage in this run (see `myrun.sh` for the full, unabridged history including dead ends), plus what needs to be set up before that command will work. `--lm_config conf/train_transformer_opt350.yaml` reproduces the paper's OPT-350M-initialized main results; drop it (or point it at `conf/train_transformer_size768_e12.yaml`) for the from-scratch ablation instead.

| Stage | Command | Setup required first |
|---|---|---|
| 1. Data prep | `./run.sh --stage 1 --stop_stage 1 --local_data_opts "--stage 2"` | `db.sh` corpus paths set up (symlinks or fresh download, see above). To reproduce the paper's D_Bal/D_3M/D_Set scale instead of the full corpus, additionally run `./local/data.sh --stage 2 --stop_stage 2 --data_config bal` (or `3m`/`set`) afterward — this re-subsamples `speechlm`/`textlm` train down to the paper's sizes without touching ASR/TTS data. |
| 2. Wav dump | `./run.sh --stage 2 --stop_stage 2` | `resampy` and `h5py` installed in the espnet conda env (TTS resampling / kmeans feature dumping import them and aren't in the base install). |
| 3. Kmeans + labeling | `scripts/feats/perform_kmeans.sh --stage 2 --stop-stage 2 --train_set train --dev_set dev --other_sets test --datadir dump/audio_raw/kmeans_pool --featdir dump/extracted/kmeans_pool --audio_format flac --feature_type hubert_base --layer 6 --feature_conf '{type=s3prl,conf={s3prl_conf={upstream=hubert_base},download_dir=ckpt,multilayer_feature=False,layer=6}}' --km_dir exp/kmeans/hubert_base_6_1000clusters --portion 1.0 --nclusters 1000 --storage_save_mode true --use_gpu true --nj 16 --cpu_cmd "slurm.pl --config conf/slurm_kmeans.conf" --cuda_cmd slurm.pl` (kmeans training only), then `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ./run.sh --stage 3 --stop_stage 3 --learn_kmeans false` (labeling asr/tts/speechlm) | `learn_kmeans.py` needs ~120GB RAM to fit on the full pool — the default cpu/gpu partitions only give ~16-64GB per job, so this step is routed to `conf/slurm_kmeans.conf` (pins it to a `gpu-a100` node for the RAM, no GPU actually used). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps avoid GPU-memory-fragmentation OOM during labeling on a shared/contended GPU. |
| 4. Assemble multitask text | `./run.sh --stage 4 --stop_stage 4` | none |
| 5. BPE training | `./run.sh --stage 5 --stop_stage 5` | none |
| 6. Collect stats | `./run.sh --stage 6 --stop_stage 6` | none — but see the Stage 7 note below: the resulting `exp/lm_stats_*/train/text_shape.bpe` needs a length filter before training, or self-attention memory blows up. |
| 7. LM training | `export HF_HOME=$PWD/hf_cache; export HF_HUB_OFFLINE=1; ./run.sh --stage 7 --stop_stage 7 --ngpu 1 --lm_config conf/train_transformer_opt350.yaml` | **cmd.sh**: `cuda_cmd` pointed at `conf/slurm_clsp_a100.conf` (needs the full 80GB A100 for this model/batch size). **OPT weights**: since `gpu-a100` compute nodes have no internet, pre-download `facebook/opt-350m` from a login node into a local `HF_HOME` first (`python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/opt-350m')"` with `HF_HOME` set to the same path used above), then train with `HF_HUB_OFFLINE=1` so it never tries the network. **Sequence-length filter**: self-attention memory is O(L²), so `batch_type: length`'s `batch_bins` (a simple linear token-count budget) badly underestimates memory for long sequences — filter `exp/lm_stats_*/train/text_shape.bpe` down to length ≤1500 (drops ~0.4% of utterances) before training, or it OOMs on a full A100 even with `batch_bins` as low as 10000. `dev`/`valid` shapes don't need this (naturally short). |
| 8. Perplexity (textLM/speechLM) | `./run.sh --stage 8 --stop_stage 8 --ngpu 1 --lm_config conf/train_transformer_opt350.yaml --inference_lm valid.acc.best.pth` | Explicit `--inference_lm valid.acc.best.pth`: the default `valid.acc.ave.pth` is only generated once training reaches `max_epoch` or an (unconfigured) early-stopping trigger, so unless you let training run to completion it won't exist. This stage is a single forward pass (`batch_size=1`, fp32) — it does **not** need an A100; see the GPU-partition note below if you want to run it on a smaller/regular GPU. |
| 9-10. ASR/TTS decoding | `./run.sh --stage 9 --stop_stage 10 --ngpu 1 --lm_config conf/train_transformer_opt350.yaml --inference_lm valid.acc.best.pth` | Same `--inference_lm` note as Stage 8. `sclite` (via `path.sh`) is used for ASR WER scoring. Stage 10 (TTS) only produces discrete-unit token sequences (`hyp.tok`) — turning those into audio for CER/MOSNet needs a matching HiFiGAN unit-vocoder, which this recipe does not include. |

**Running Stage 8-10 on a non-A100 GPU** (e.g. if your A100 allocation/quota is exhausted): these stages are light enough for any GPU (verified on an 11GB GTX 1080 Ti). Since `lm.sh` re-sources `cmd.sh` internally, you can't override `cuda_cmd` with a plain `export`; instead temporarily point it at the regular `gpu` partition and back, e.g.:

```bash
trap 'sed -i "s#cuda_cmd=\"slurm.pl --config conf/slurm.conf\"#cuda_cmd=\"slurm.pl --config conf/slurm_clsp_a100.conf\"#" cmd.sh' EXIT
sed -i 's#cuda_cmd="slurm.pl --config conf/slurm_clsp_a100.conf"#cuda_cmd="slurm.pl --config conf/slurm.conf"#' cmd.sh
./run.sh --stage 8 --stop_stage 8 --ngpu 1 --lm_config conf/train_transformer_opt350.yaml --inference_lm valid.acc.best.pth
```

the `trap` guarantees `cmd.sh` is switched back to the A100 config on exit (success, failure, or interrupt) so a later Stage 7 resume doesn't silently run on too little GPU memory.

**Reporting ASR WER split by test-clean/test-other** (the paper reports these separately, but Stage 9's `sclite` run scores the combined `test` set): re-score the existing `hyp.trn`/`ref.trn` from Stage 9 without re-decoding, using the utterance-ID lists from `data/librispeech/asr/test_clean` and `test_other` (the pre-`combine_data.sh` intermediates) — see the "13." block near the end of `myrun.sh` for the exact commands.

### Known deviations from the paper's full eval suite

- **sWUGGY / sBLIMP** (lexical/syntactic zero-shot probing): not implemented anywhere in ESPnet. The official data is the `sLM21` benchmark package (`https://download.zerospeech.com/datasets/sLM21.dataset.zip`, ~30.6GB, no registration needed) plus the `zerospeech-benchmarks` (`zrc`) scoring toolkit; both `lexical/` (sWUGGY) and `syntactic/` (sBLIMP) items ship with paired text (word/sentence) alongside audio, so both the textLM and speechLM tracks can be evaluated from this one download. Not yet wired up to this recipe.
- **TTS CER / MOSNet**: require converting Stage 10's discrete-unit output into actual audio via a HiFiGAN vocoder (+ x-vector speaker embedding, per the paper), then an external ASR (CER) and MOSNet (quality). None of this is implemented here yet — Stage 10 only gets as far as the unit sequences.
