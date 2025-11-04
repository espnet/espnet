# Emilia Recipe
This recipe is for training a text-to-speech model on the Emilia dataset. For simplicity, we use only a subset of the Emilia English dataset (approximately 100 hours), and the trained model is tested on the VCTK test dataset. The default model for this recipe is a multi-speaker (x-vectors) VITS model.

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