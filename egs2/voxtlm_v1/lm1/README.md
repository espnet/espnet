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
