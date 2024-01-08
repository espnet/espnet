# OWSM v3.1

OWSM v3.1 is an improved version of OWSM v3. We do not include any new training data. Instead, we adopt a state-of-the-art speech encoder, [E-Branchformer](https://arxiv.org/abs/2210.00077).

Compared to v3, here we do not use WSJ for training. Also, the text transcriptions of some datasets are in all upper case, which are converted to lower case.

- Model: [https://huggingface.co/espnet/owsm_v3.1_ebf](https://huggingface.co/espnet/owsm_v3.1_ebf)


### Results

Here are some of the ASR results (WER or CER) with attention-based greedy search.

|Test set|Language|Whisper medium (769M)|OWSM v3 (889M)|OWSM v3.1 (1.02B)|
|:---:|:---:|:---:|:---:|:---:|
|LibriSpeech test-clean|eng|2.8|2.7|2.4|
|LibriSpeech test-other|eng|6.5|6.0|5.0|
|Switchboard eval2000|eng|19.4|17.2|16.3|
|TEDLIUM|eng|5.1|4.8|5.1|
|WSJ eval92|eng|2.9|13.4|3.5|
|CommonVoice|eng|11.9|14.5|12.6|
|FLEURS|eng|6.4|10.9|9.0|
|VoxPopuli|eng|7.6|9.2|8.4|
|Multilingual LibriSpeech|eng|10.2|7.4|7.1|
|Multilingual LibriSpeech|spa|6.1|11.7|9.0|
|Multilingual LibriSpeech|fra|9.7|14.1|12.1|
|Multilingual LibriSpeech|deu|8.1|11.9|10.8|
|Multilingual LibriSpeech|nld|12.2|17.7|18.1|
|Multilingual LibriSpeech|ita|15.6|24.5|20.2|
|Multilingual LibriSpeech|por|8.9|28.2|21.6|
|Multilingual LibriSpeech|pol|6.8|37.0|25.2|
|AISHELL-1|zho|15.7|7.1|6.4|
|ksponspeech eval-clean|kor|17.6|20.5|16.7|
|ksponspeech eval-other|kor|12.8|22.6|18.9|
|ReazonSpeech|jpn|25.3|11.3|7.9|


### Guidance for data preparation
1. Please work progressively from v1 to v3: this means you need to prepare data for v1, v2 and v3 in order to obtain the full v3 data. To start the data preparation, run `bash local/data.sh --VERSION v1 # or v2, v3`
2. Please revise `db.sh` for all datasets before running `local/data.sh`. Some datasets cannot be downloaded and untared automatically due to license issues. Users should take care of it by themselves.
3. Due to the large volume of data, we are not confident the scripts will run smoothly for each dataset. Please raise an issue if you believe there is a bug.
4. This script only prepares data for train and valid subsets. Test data should be prepared separately following the conventional ESPnet2 format.
5. Even though we provide this centralized data preparation script and combine all datasets in it, we strongly recommend users to NOT use the merged train_v* and valid_v* for feature extractions. Instead, users may run stage 2-4 for each dataset separately and combine all datasets together under `dump/raw` directory. This will allow you to handle all datasets simultaneously; inspection and debugging will also be easier. This is exactly what we did in our experiments.
6. Users can also refer to this PR to check more details: https://github.com/espnet/espnet/pull/5478
7. The detailed data list is in `local/data.sh`. Also see: https://arxiv.org/pdf/2309.13876.pdf
