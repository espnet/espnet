# QASR-TTS RECIPE

- Our goal is to build a TTS character based system using semi-supervised data selection in a low-resource scenario by proposing two different methodologies:
the first one by training non-autoregressive (non-AR) model from scratch on very small amount of data; and the second one by training
autoregressive (AR) model finetuned on top of a pre-trained model.

- Step 1: Prepare the data

- Step 2: Download a pretrained model

- Step 3: Replace token list with the pretrained model's one

- Step 4: Finetune the pre-trained model on our 1 hour dataset and we excluded the embedding layer since we are finetuning on a different language

- Step 5: Use the finetuned model as a teacher model to train the Non-AR model FastSpeech2

- Step 6: Train Parallel Wav GAN model to produce better wav samples

Go to this link to download the dataset: https://arabicspeech.org/qasr_tts
