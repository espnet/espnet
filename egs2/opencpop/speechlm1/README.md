# This is ESPnet2 Speech Language Model (SpeechLM) Recipe
🎙️ ``Task``: Singing Voice Synthesis (SVS)

📊 ``Corpus``: Mandarin Single Female Singer Corpus, [Opencpop](https://wenet.org.cn/opencpop/download/).

Some Tutorials
- [PSC usage tutorial](https://www.wavlab.org/activities/2022/psc-usage/)
- [Espnet recipe tutorial](https://github.com/espnet/notebook/blob/master/ESPnet2/Course/CMU_SpeechRecognition_Fall2022/recipe_tutorial.ipynb)
- [Speechlm Template](https://github.com/espnet/espnet/blob/speechlm/egs2/TEMPLATE/speechlm1/README.md#stage-10-evaluation)

Please make sure you have installed ESPnet and [VERSA](https://github.com/shinjiwlab/versa/).


## Data Structure
* Download this corpus from the [official websites](https://wenet.org.cn/opencpop/download/)

* Organize the data as
    ```
    opencpop/speechlm1
        |__downloads
        |   |__raw_data____midis
        |   |           |__textgrids
        |   |           |__wavs
        |   |__segments____wavs
        |               |__test.txt
        |               |__train.txt
        |               |__transcriptions.txt
        |__local
            |__midi-note.scp
    ```

* We use modified ``label`` as the conditional entry, ``wav.scp`` as the target entry. Information from ``text`` and ``score`` modalities of the original SVS are integrated to ``label`` through data preprocessing (stage 2). Also, the tokens from SVS are distinguished by tokens from TTS using a ``svs_`` prefix.

## Run the Recipe

Use stage1-5 for data preparation and generating token list. Please make sure you don't have overlapping validation set and test sets when running stage2, which will duplicatedly preprocess validation set and cause errors. Feel free to change the config in ``./run.sh`` after stage2.
```
./run.sh --stage 1 --stop_stage 5
```

Since this dataset is relatively small. To avoid overfitting, we use a pretrained TTS model ([download and setup](https://github.com/espnet/espnet/blob/speechlm/egs2/TEMPLATE/speechlm1/README.md#pretrained-models)). To combine the svs tokens to the pretrained token list, run

```
python pyscripts/utils/speechlm_extend_vocab.py \
  --input_token_list_dir data/token_list/tts_vocab \
  --output_token_list_dir data/token_list/tts_vocab_ext_phone \
  --input_exp_dir exp/speechlm_espnet_speechlm_pretrained_tts \
  --output_exp_dir exp/speechlm_espnet_speechlm_pretrained_tts_ext_phone \
  --inference_model 60epoch.pth \
  --additional_vocabs dump/raw_svs_opencpop/tr_no_dev/token_lists/svs_lb_token_list \
  --additional_task svs
```

Skip stage 6 if using the pretrained model.

```
./run.sh --stage 7
```
