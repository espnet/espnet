## Use transfer learning for ASR in ESPnet2

In that tutorial, we will introduce several options to use pre-trained models/parameters for Automatic Speech Recognition (ASR) in ESPnet. Available options are :
- use a local model you (or a collegue) have already trained,
- use a trained model from ESPnet repository on HuggingFace.

We note that this is done for ASR training, so at __stage 11__ of ESPnet2 models' recipe.

### 0. Why using such (pre-)trained models ?

Several projects may involve making use of previously trained models, this is the reason why we developed ESPnet repository on HuggingFace for instance.
Example of use cases are listed below (non-exhaustive):
- target a low resource language, a model trained from scratch may perform badly if trained with only few hours of data,
- study robustness to shifts (domain, language ... shifts) of a model,
- use massively trained multilingual ASR models ...

### 1. Use a local model that you have already trained.

__Step 1__ : make sure your ASR model file has the proper ESPnet format (should be ok if trained with ESPnet). It just needs to be a ".pth" (or ".pt" or other extension) type pytorch model.

__Step 2__ : add the parameter ```--pretrained_model path/to/your/pretrained/model/file.pth``` to run.sh.

__Step 3__ : step 2 will initialize your new model with the parameters of the pre-trained model. Thus your new model will be trained with a strong initialization. However, if your new model has different parameter sizes for some parts of the model (e.g. last projection layer could be modified ...). This will lead to an error because of mismatches in size. To prevent this to happen, you can add the parameter ```--ignore_init_mismatch true``` in run.sh.

__Step 4 (Optional)__ : if you only want to use some specific parts of the pre-trained model, or exclude specific parts, you can specify it in the ```--pretrained_model``` argument by passing the component names with the following syntax : ```--pretrained_model <file_path>:<src_key>:<dst_key>:<exclude_Keys>```. ```src_key``` are the parameters you want to keep from the pre-trained model. ```dst_key``` are the parameters you want to initialize in the new model with the ```src_key```parameters. And ```exclude_Keys``` are the parameters from the pre-trained model that you do not want to use. You can leave ```src_key``` and ```dst_key``` fields empty and just fill ```exclude_Keys``` with the parameters that you want to drop. For instance, if you want to re-use encoder parameters but not decoder ones, syntax will be ```--pretrained_model <file_path>:::decoder```.  You can see the argument expected format in more details [here](https://github.com/espnet/espnet/blob/e76c78c0c661ab37cc081d46d9b059dcb31292fe/espnet2/torch_utils/load_pretrained_model.py#L43-L53).

__Additional note about the ```--ignore_init_mismatch true``` option :__ This option is very convenient because in lots of transfer learning use cases, you will aim to use a model trained on a language X (e.g. X=English) for another language Y. Language Y may have a vocabulary (set of tokens) different from language X, for instance if you target Y=Totonac, a Mexican low resource language, your model may be stronger if you use a different set of bpes/tokens that the one used to train the English model. In that situation, the last layer (projection to vocabulary space) of your ASR model needs to be initialized from scratch and may be different in shape than the one of the English model. For that reason, you should use the ```--ignore_init_mismatch true``` option. It also enables to handle the case where the scripts are differents from languages X to Y.


### 2. Use a trained model from ESPnet repository on HuggingFace.

[ESPnet repository on HuggingFace](https://huggingface.co/espnet) contains more than 200 pre-trained models, for a wide variety of languages and dataset, and we are actively expanding this repositories with new models every week! This enables any user to perform transfer learning with a wide variety of models without having to re-train them.
In order to use our pre-trained models, the first step is to download the ".pth" model file from the [HugginFace page](https://huggingface.co/espnet). There are several easy ways to do it, either by manually downloading them (e.g. ```wget https://huggingface.co/espnet/bn_openslr53/blob/main/exp/asr_train_asr_raw_bpe1000/41epoch.pth```), cloning it (```git clone https://huggingface.co/espnet/bn_openslr53```) or downloading it through an ESPnet recipe (described in the models' pages on HuggingFace):
```cd espnet
git checkout fa1b865352475b744c37f70440de1cc6b257ba70
pip install -e .
cd egs2/bn_openslr53/asr1
./run.sh --skip_data_prep false --skip_train true --download_model espnet/bn_openslr53
```

Then, as you have the ".pth" model file, you can follow the steps 1 to 4 from the previous section in order to use this pre-train model.
