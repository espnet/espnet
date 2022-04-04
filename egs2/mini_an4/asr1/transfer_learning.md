## Use transfer learning or Self-Supervised pre-trained models for ASR in ESPnet

In that tutorial, we will introduce several options to use pre-trained models/parameters for Automatic Speech Recognition (ASR) in ESPnet. Available options are : 
- use a local model you (or a collegue) have already trained,
- use a trained model from ESPnet repository on HuggingFace,
- use a Self-Supervised pre-trained model (e.g. Hubert, Wav2Vec2, WavLM ...).


### 0. Why using such (pre-)trained models ? 

Several projects may involve making use of previously trained models, this is the reason why we developed ESPnet repository on HuggingFace for instance.
Example of use cases are listed below (non-exhaustive):
- target a low resource language, a model trained from scratch may perform badly if trained with only few hours of data,
- study robustness to shifts (domain, language ... shifts) of a model,
- use state of the art Self-Supervised models for any purpose (low resource, medium resource, semi-supervised adaptations ...) ...

### 1. Use a local model that you have already trained 

Step 1 : make sure your ASR model folder has the proper ESPnet format (should be ok if trained with ESPnet). It just needs to be a ".pth" type pytorch model.

Step 2 : add the parameter ```--pretrained_model path/to/your/pretrained/model/file.pth``` to run.sh. 

Step 3: step 2 will initialize your new model with the parameters of the trained model. Thus your new model will be trained with a strong initialization. However, if your new model have different parameter sizes for some parts of the model (e.g. last projection layer could be modified ...). This will lead to an error because of mismatches in size. To prevent this to happen, you can add the parameter ```--ignore_init_mismatch true```in run.sh.

Step 4 (Optional): if you only want to use some specific parts of the trained model, or exclude specific parts, you can specify it in the . You can see the argument expected format in more details [here](https://github.com/espnet/espnet/blob/e76c78c0c661ab37cc081d46d9b059dcb31292fe/espnet2/torch_utils/load_pretrained_model.py#L43-L53).
