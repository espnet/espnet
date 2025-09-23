# README

This language identification model was trained using the ESPnet recipe from [ESPnet](https://github.com/espnet/espnet/) toolkit. It leverages the pretrained [MMS-1B](https://huggingface.co/facebook/mms-1b) as the encoder and [ECAPA-TDNN](https://arxiv.org/pdf/2005.07143) as the embedding extractor for robust spoken language identification.

The model is trained on the [VoxLingua107](https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/) dataset, which comprises over 6,600 hours of speech spanning 107 languages. Speech segments are sourced from YouTube videos and annotated using metadata.

This recipe provides data preparation scripts and training setup.

<details>
<summary>Supported languages</summary>

- abk
- afr
- amh
- ara
- asm
- aze
- bak
- bel
- ben
- bod
- bos
- bre
- bul
- cat
- ceb
- ces
- cmn
- cym
- dan
- deu
- ell
- eng
- epo
- est
- eus
- fao
- fas
- fin
- fra
- glg
- glv
- grn
- guj
- hat
- hau
- haw
- heb
- hin
- hrv
- hun
- hye
- ina
- ind
- isl
- ita
- jav
- jpn
- kan
- kat
- kaz
- khm
- kor
- lao
- lat
- lav
- lin
- lit
- ltz
- mal
- mar
- mkd
- mlg
- mlt
- mon
- mri
- msa
- mya
- nep
- nld
- nno
- nor
- oci
- pan
- pol
- por
- pus
- ron
- rus
- san
- sco
- sin
- slk
- slv
- sna
- snd
- som
- spa
- sqi
- srp
- sun
- swa
- swe
- tam
- tat
- tel
- tgk
- tgl
- tha
- tuk
- tur
- ukr
- urd
- uzb
- vie
- war
- yid
- yor

</details>

## Usage

To train the model from scratch, simply run:

```bash
./run.sh
```

For using pretrained checkpoints, please see the Hugging Face repository [espnet/lid_voxlingua107_mms_ecapa](https://huggingface.co/espnet/lid_voxlingua107_mms_ecapa).

## Train and Evaluation Datasets

The training used only the VoxLingua107 dataset, comprising 6,628 hours of speech across 107 languages from YouTube.

| Dataset       | Domain      | #Langs. Train/Test | Dialect | Training Setup (VL107-only) |
| ------------- | ----------- | ------------------ | ------- | --------------------------- |
| [VoxLingua107](https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/)  | YouTube     | 107/33             | No      | Seen                        |
| [Babel](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=31a13cefb42647e924e0d2778d341decc44c40e9)         | Telephone   | 25/25              | No      | Unseen                      |
| [FLEURS](https://huggingface.co/datasets/google/xtreme_s)        | Read speech | 102/102            | No      | Unseen                      |
| [ML-SUPERB 2.0](https://huggingface.co/datasets/espnet/ml_superb_hf) | Mixed       | 137/(137, 8)       | Yes     | Unseen                      |
| [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli)     | Parliament  | 16/16              | No      | Unseen                      |


## Results

**Accuracy (%) on In-domain and Out-of-domain Test Sets**

<style>
.hf-model-cell {
    max-width: 120px;
    overflow-x: auto;
    white-space: nowrap;
    scrollbar-width: thin;
    scrollbar-color: #888 #f1f1f1;
}

.config-cell {
    max-width: 100px;
    overflow-x: auto;
    white-space: nowrap;
    scrollbar-width: thin;
    scrollbar-color: #888 #f1f1f1;
}

.hf-model-cell::-webkit-scrollbar,
.config-cell::-webkit-scrollbar {
    height: 6px;
}

.hf-model-cell::-webkit-scrollbar-track,
.config-cell::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 3px;
}

.hf-model-cell::-webkit-scrollbar-thumb,
.config-cell::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 3px;
}

.hf-model-cell::-webkit-scrollbar-thumb:hover,
.config-cell::-webkit-scrollbar-thumb:hover {
    background: #555;
}
</style>

<div style="overflow-x: auto;">

| Config                    | 🤗 HF Repo | VoxLingua107 | Babel | FLEURS | ML-SUPERB2.0 Dev | ML-SUPERB2.0 Dialect | VoxPopuli | Macro Avg. |
| ------------------------- | ----------- | ------------ | ----- | ------ | ---------------- | -------------------- | --------- | ---------- |
| <div class="config-cell">`conf/mms_ecapa_baseline`</div> | <div class="hf-model-cell">[espnet/lid_voxlingua107_mms_ecapa](https://huggingface.co/espnet/lid_voxlingua107_mms_ecapa)</div> | 94.2         | 86.7  | 95.8   | 89.0             | 73.4                 | 85.6      | 87.5       |

</div>


**Note:**

The recommended transformers version is 4.51.3, and s3prl is 0.4.17.
