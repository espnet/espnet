# README

This recipe provides a spoken language identification (LID) setup using the VoxLingua107 dataset, which contains over 6600 hours of speech in 107 languages. The speech segments are extracted from YouTube videos and labeled based on metadata, with a validated development set covering 33 languages.



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
