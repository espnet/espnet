<!-- Generated by scripts/utils/show_asr_result.sh -->
# ML_SUPERB data download
This can be referred to the ML-SUPERB [recipe](https://github.com/espnet/espnet/blob/master/egs2/ml_superb/asr1)

[Download link](https://drive.google.com/file/d/1QYjl-7vflle__3AfuosAC5VJGiBDvEqz/view?usp=drive_link)

After download the dataset, please set the `MLSUPERB` to the data directory. The preparation will be automatically done in scripts for each tasks.

# Bitrate calculation

We show how to calculate the bitrate in the baseline system (16.49). For each eval set, it follows three steps:
1. Convert the input text to index (`token_int.rm.wavlm_large_21_km2000`). This is because the baseline uses subword modeling (e.g BPE), and it is represented in CJK characters. **Note: as long as the input text is in index, this step can be skipped.**
   ```
   1272-135031-0014        784 0 1867 1102 866 2042 1209 184 187 1209 768 3 17 111 940 202 69 1518 217 337 1984 121 2383 916 276 2470 1287 217 833 508 690 161 211 161 211 206 174 30 96 3 729
   ```
2. `pyscripts/utils/convert_token2json.py` generates json file for `vocab`, `tokens`, `ref_len`.
3. `pyscripts/utils/calculate_bitrate.py` computes the bitrate

Finally, the overall bitrate is computed by aggregating the results of all eval sets.

```bash
$ token_suffix="wavlm_large_21_km2000"
$ bpe_folder="data/token_list/src_bpe_unigram3000_rm_wavlm_large_21_km2000"
$ bitrate_dir="./bitrate"
$ for dset in dev_clean dev_other test_clean test_other test_1h; do
$   paste \
$     <(<dump/raw/${dset}/text.rm.${token_suffix} cut -d" " -f1) \
$     <(<dump/raw/${dset}/text.rm.${token_suffix} spm_encode --model=${bpe_folder}/bpe.model --output_format=id) \
$     > dump/raw/${dset}/token_int.rm.${token_suffix}

$   python pyscripts/utils/convert_token2json.py \
$     --vocab data/token_list/src_bpe_unigram3000_rm_${token_suffix}/tokens.txt \
$     --token dump/raw/${dset}/token_int.rm.${token_suffix} \
$     --ref_scp data/${dset}/wav.scp \
$     --result_dir "${bitrate_dir}/${dset}"

$   python pyscripts/utils/calculate_bitrate.py \
$     --vocab "${bitrate_dir}/${dset}"/vocab.json \
$     --tokens "${bitrate_dir}/${dset}"/tokens.json \
$     --reference_len "${bitrate_dir}/${dset}"/ref_len.scp \
$     --bitrate_details "${bitrate_dir}/${dset}"/details.txt
$ done

$ python - <<EOF
import numpy as np
bitrates=[]
bitrate_dir="${bitrate_dir}"
for dset in ["dev_clean", "dev_other", "test_clean", "test_other", "test_1h"]:
    with open(f"{bitrate_dir}/{dset}/details.txt", "r") as f:
        for line in f.readlines():
            lst = line.strip().split()
            bitrates.append(float(lst[1]))
print(np.round(np.mean(bitrates), 2))

EOF

$ 16.49
```

# RESULTS
## Environments
- date: `Wed Jan 17 08:22:49 EST 2024`
- python version: `3.9.13 (main, Aug 25 2022, 23:26:10)  [GCC 11.2.0]`
- espnet version: `espnet 202310`
- pytorch version: `pytorch 1.13.1`
- Git hash: `0d77ccfd8d980a996ac821253234a67a15f63129`
  - Commit date: `Mon Oct 30 15:19:44 2023 -0400`

## Ebranchformer_wavlm_large_21_km2000_bpe_rm3000_bpe_ts6000
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_clean|2703|54402|95.9|3.9|0.2|0.4|4.5|48.2|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_other|2864|50948|92.5|6.9|0.6|0.6|8.1|60.4|
|decode_ctc0.3_asr_model_valid.acc.ave/test_1h|7439|57426|14.5|61.3|24.2|14.8|100.3|98.0|
|decode_ctc0.3_asr_model_valid.acc.ave/test_clean|2620|52576|96.0|3.8|0.3|0.4|4.4|47.6|
|decode_ctc0.3_asr_model_valid.acc.ave/test_other|2939|52343|92.4|7.0|0.6|0.6|8.3|63.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_clean|2703|288456|98.9|0.7|0.5|0.4|1.5|48.2|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_other|2864|265951|97.5|1.4|1.0|0.7|3.2|60.4|
|decode_ctc0.3_asr_model_valid.acc.ave/test_1h|7439|299326|44.4|28.4|27.2|17.0|72.6|98.0|
|decode_ctc0.3_asr_model_valid.acc.ave/test_clean|2620|281530|98.9|0.6|0.5|0.4|1.4|47.6|
|decode_ctc0.3_asr_model_valid.acc.ave/test_other|2939|272758|97.6|1.4|1.0|0.7|3.1|63.0|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_clean|2703|82834|95.2|3.5|1.3|0.5|5.3|48.2|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_other|2864|76205|91.6|6.4|2.0|1.1|9.5|60.4|
|decode_ctc0.3_asr_model_valid.acc.ave/test_1h|7439|159974|26.2|48.4|25.4|15.0|88.8|98.0|
|decode_ctc0.3_asr_model_valid.acc.ave/test_clean|2620|81195|95.6|3.2|1.2|0.5|4.9|47.6|
|decode_ctc0.3_asr_model_valid.acc.ave/test_other|2939|78676|91.6|6.2|2.2|1.0|9.5|63.0|
