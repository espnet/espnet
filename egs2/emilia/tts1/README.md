# Emilia Recipe
This recipe trains a text-to-speech (TTS) model on the Emilia dataset. Since the Emilia dataset does not include a test set, the VCTK dataset is used as the test set. The default model for this recipe is a multi-speaker VITS model that uses x-vectors.

For more details on how to run this recipe, please refer to the following page:
- [How to run the recipe](../../TEMPLATE/tts1/README.md#how-to-run)

# Prerequisite
Since downloading the Emilia dataset requires Hugging Face authentication, you must first accept the conditions on the [Hugging Face Emilia dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset) page. Then, add your Hugging Face token as an environment variable:
```bash
export HF_TOKEN=${YOUR_TOKEN}
```
After that, you are ready to go!

# Downloading a Subset of the Emilia Dataset
The Emilia dataset is quite large, containing 46.8k hours of speech. If you want to download only a subset of the dataset, modify the `START_INDEX` and `END_INDEX` values in `local/data_download.sh`. For example, to download approximately 100 hours of data, you can set `START_INDEX` and `END_INDEX` to 0 and 1, respectively.

# Creating Kaldi-Format Files for the Emilia Dataset
Because the Emilia dataset is very large, creating Kaldi-format files may take a long time. To speed up the process, the code supports parallel processing, allowing you to specify the number of jobs to run in parallel:

```bash
./run.sh --stage 1 --stop-stage 1 --local_data_opts "--nj 32"
```
where `nj` specifies the number of parallel jobs.

# Changing the Language
The Emilia dataset includes six languages: English (EN), Chinese (ZH), German (DE), French (FR), Japanese (JA), and Korean (KO). You can specify the language to train with:

```bash
./run.sh --stage 1 --stop-stage 1 --local_data_opts "--lang ZH"
```

The `--lang` argument must use one of the short codes listed above. When set, the dataset download and preprocessing steps will run for that language only. If not specified, the default language is English.

# Results
## Environments
- date: `Wed Mar 25 19:56:06 EDT 2026`
- python version: `3.11.14`
- espnet version: `espnet 202511`
- Git hash: `fe549eb691e79a9906956afaeee155faa97ab170`
  - Commit date: `Thu Mar 26 11:10:35 EDT 2026`

Epoch 700
| Metric | Value |
|:--|--:|
| UTMOS | 2.69 |
| MCD | 17.65 |
| DNS (overall) | 3.07 |
| DNS (P.808) | 3.62 |
| Whisper WER | 0.35 |
| Whisper CER | 0.22 |
| SingMOS | 3.71 |
| Speaker similarity | 0.35 |

The model can be accessed at: [https://huggingface.co/NewGame/Emilia-vits-espnet2](https://huggingface.co/NewGame/Emilia-vits-espnet2)
