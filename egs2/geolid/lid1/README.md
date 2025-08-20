# README

This geolocation-aware language identification (LID) model is developed using the [ESPnet](https://github.com/espnet/espnet/) toolkit. It integrates the powerful pretrained [MMS-1B](https://huggingface.co/facebook/mms-1b) as the encoder and employs [ECAPA-TDNN](https://arxiv.org/pdf/2005.07143) as the embedding extractor to achieve robust spoken language identification.

The main innovations of this model are:
1. Incorporating geolocation prediction as an auxiliary task during training.
2. Conditioning the intermediate representations of the self-supervised learning (SSL) encoder on intermediate-layer information.
This geolocation-aware strategy greatly improves robustness, especially for dialects and accented variations.

For further details on the geolocation-aware LID methodology, please refer to our paper: *Geolocation-Aware Robust Spoken Language Identification* (arXiv link to be added).

## Usage

### Prerequisites

First, ensure you have ESPnet installed. If not, follow the [ESPnet installation instructions](https://espnet.github.io/espnet/installation.html).

This project requires **modified versions** of s3prl and transformers for geolocation conditioning functionality.

**Install Modified s3prl:**
```bash
# If you have already installed s3prl, please uninstall it first
pip uninstall s3prl  # (Optional if already installed)

# Clone and install the modified version
git clone -b lid https://github.com/Qingzheng-Wang/s3prl.git
cd s3prl
pip install -e .
cd ..
```

**Install Modified Transformers:**
```bash
# If you have already installed transformers, please uninstall it first
pip uninstall transformers  # (Optional if already installed)

# Clone and install the modified version
git clone -b v4.51.3-qingzheng https://github.com/Qingzheng-Wang/transformers.git
cd transformers
pip install -e .
cd ..
```

⚠️ **Important:** Make sure to use the exact versions specified above for compatibility.

### Quick Start

**Step 1: Navigate to project directory**
```bash
cd espnet/egs2/geolid/lid1
```

If you would like to train the model or run inference using our prepared datasets (instead of your own), please first follow the [data preparation instructions](local/README.md) to set up the required data.

**Option 1: Train with Combined Dataset**
```bash
# Uses all available datasets for training
./run_combined.sh --stage 4 --stop_stage 8
```

**Option 2: Train with VoxLingua107 Only**
```bash
# Four different configurations available
# Check run_voxlingua107_only.sh for configuration details
./run_voxlingua107_only.sh --stage 4 --stop_stage 8
```

We also provide pre-trained models for immediate use:

```bash
# Download the exp_combined to egs2/geolid/lid1
hf download espnet/geolid_vl107only_shared_trainable --local-dir . --exclude "README.md" "meta.yaml" ".gitattributes"

./run_voxlingua107_only.sh --skip_data_prep false --skip_train true --lid_config conf/voxlingua107_only/mms_ecapa_upcon_32_44_it0.4_shared_trainable.yaml --stage 6 --stop_stage 8

```

To run inference with checkpoints trained on the combined training set, please use the script `run_combined.sh`.

If you want to perform inference on your own prepared datasets, modify the test_sets field in the corresponding run script.

See the [Results](#results) section below for detailed performance metrics and Hugging Face model links.

## Train and Evaluation Datasets

Two training setups are provided:
1. VoxLingua107-Only: The training used only the VoxLingua107 dataset, comprising 6,628 hours of speech across 107 languages from YouTube.
2. Combined: The training utilized a combined dataset, merging five domain-specific corpora, resulting in 9,865 hours of speech data covering 157 languages.

Below is a summary table of the training and evaluation datasets used in this project:

| Dataset                                                      | Domain      | #Langs. Train/Test | Dialect | Training Setup (VL107-only) | Training Setup (Combined) |
| ------------------------------------------------------------ | ----------- | ------------------ | ------- | --------------------------- | ------------------------- |
| [VoxLingua107](https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/) | YouTube     | 107/33             | No      | Seen                        | Seen                      |
| [Babel](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=31a13cefb42647e924e0d2778d341decc44c40e9) | Telephone   | 25/25              | No      | Unseen                      | Seen                      |
| [FLEURS](https://huggingface.co/datasets/google/xtreme_s)    | Read speech | 102/102            | No      | Unseen                      | Seen                      |
| [ML-SUPERB 2.0](https://huggingface.co/datasets/espnet/ml_superb_hf) | Mixed       | 137/(137, 8)       | Yes     | Unseen                      | Seen                      |
| [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) | Parliament  | 16/16              | No      | Unseen                      | Seen                      |


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
| <div class="config-cell">`conf/voxlingua107_only/mms_ecapa_upcon_32_44_it0.4_independent_frozen.yaml`</div> | <div class="hf-model-cell">[espnet/geolid_vl107only_independent_frozen](https://huggingface.co/espnet/geolid_vl107only_independent_frozen)</div> | 94.2         | 87.1  | 95.0   | 89.0             | 77.2                 | 90.4      | 88.8       |
| <div class="config-cell">`conf/voxlingua107_only/mms_ecapa_upcon_32_44_it0.4_independent_trainable.yaml`</div> | <div class="hf-model-cell">[espnet/geolid_vl107only_independent_trainable](https://huggingface.co/espnet/geolid_vl107only_independent_trainable)</div> | 93.7         | 85.3  | 93.7   | 88.3             | 70.3                 | 86.5      | 86.3       |
| <div class="config-cell">`conf/voxlingua107_only/mms_ecapa_upcon_32_44_it0.4_shared_frozen.yaml`</div> | <div class="hf-model-cell">[espnet/geolid_vl107only_shared_frozen](https://huggingface.co/espnet/geolid_vl107only_shared_frozen)</div> | 94.3         | 85.9  | 94.3   | 88.8             | 80.7                 | 89.2      | 88.8       |
| <div class="config-cell">`conf/voxlingua107_only/mms_ecapa_upcon_32_44_it0.4_shared_trainable.yaml`</div> | <div class="hf-model-cell">[espnet/geolid_vl107only_shared_trainable](https://huggingface.co/espnet/geolid_vl107only_shared_trainable)</div> | 94.9         | 87.7  | 93.5   | 89.3             | 78.8                 | 89.5      | 88.9       |
| <div class="config-cell">`conf/combined/mms_ecapa_upcon_32_44_it0.4_shared_trainable.yaml`</div> | <div class="hf-model-cell">[espnet/geolid_combined_shared_trainable](https://huggingface.co/espnet/geolid_combined_shared_trainable)</div> | 94.4         | 95.4  | 97.7   | 88.6             | 86.8                 | 99.0      | 93.7       |


</div>


## Citation

```BibTex
@inproceedings{wang2025geolid,
  author={Qingzheng Wang, Hye-jin Shim, Jiancheng Sun, and Shinji Watanabe},
  title={Geolocation-Aware Robust Spoken Language Identification},
  year={2025},
  booktitle={Procedings of ASRU},
}

@inproceedings{watanabe2018espnet,
  author={Shinji Watanabe and Takaaki Hori and Shigeki Karita and Tomoki Hayashi and Jiro Nishitoba and Yuya Unno and Nelson Yalta and Jahn Heymann and Matthew Wiesner and Nanxin Chen and Adithya Renduchintala and Tsubasa Ochiai},
  title={{ESPnet}: End-to-End Speech Processing Toolkit},
  year={2018},
  booktitle={Proceedings of Interspeech},
  pages={2207--2211},
  doi={10.21437/Interspeech.2018-1456},
  url={http://dx.doi.org/10.21437/Interspeech.2018-1456}
}
```
