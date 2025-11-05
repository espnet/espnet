# Emilia Recipe
This recipe trains a text-to-speech (TTS) model on the Emilia dataset. Since the Emilia dataset does not include a test set, the VCTK dataset is used as the test set. The default model for this recipe is a multi-speaker VITS model that uses x-vectors.

For more details on how to run this recipe, please refer to the following page:
- [How to run the recipe](../../TEMPLATE/tts1/README.md#how-to-run)

# Prerequisite
Since downloading the Emilia dataset requires Hugging Face authentication, you must first accept the conditions on the [Hugging Face Emilia dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset) page. Then, add your Hugging Face token to `local/data_download.sh`:
```bash
# local/data_download.sh
....
HF_TOKEN="" # Put your Hugging Face token here 
....
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

