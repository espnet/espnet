# Turn taking prediction model

- Training config: [conf/train_asr_whisper_turn_taking.yaml](conf/train_asr_whisper_turn_taking.yaml)
- Inference config: [conf/decode_asr_chunk.yaml](conf/decode_asr_chunk.yaml)
- Model link: [https://huggingface.co/espnet/Turn_taking_prediction_SWBD](https://huggingface.co/espnet/Turn_taking_prediction_SWBD)

## ROC_AUC

|dataset|Continuation|Backchannel|Turn change|Interruption|Silence|Overall|
|---|---|---|---|---|---|---|
|decode_asr_chunk_asr_model_valid.loss.ave/test|93.3|89.4|90.8|91.3|95.1|92.0|

# Guidance for training and inference

- Complete data preparation setup with turn taking labels using the label annotation sequence defined in \cite{arora2025talking}
- Get turn taking label for each 40msec and downsample chunks from training and validation set such that there are roughly similar numbers of samples for each label class.
- Scripts for preparing data for switchboard provided in this recipe.
-



# Citing ESPnet

```BibTex

@inproceedings{
arora2025talking,
title={Talking Turns: Benchmarking Audio Foundation Models on Turn-Taking Dynamics},
author={Siddhant Arora and Zhiyun Lu and Chung-Cheng Chiu and Ruoming Pang and Shinji Watanabe},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=2e4ECh0ikn}
}

@inproceedings{watanabe2018espnet,
  author={Shinji Watanabe and Takaaki Hori and Shigeki Karita and Tomoki Hayashi and Jiro Nishitoba and Yuya Unno and Nelson Yalta and Jahn Heymann and Matthew Wiesner and Nanxin Chen and Adithya Renduchintala and Tsubasa Ochiai},
  title={{ESPnet}: End-to-End Speech Processing Toolkit},
  year={2018},
  booktitle={Proceedings of Interspeech},
  pages={2207--2211},
  doi={10.21437/Interspeech.2018-1456},
  url={http://dx.doi.org/10.21437/Interspeech.2018-1456}
}

```
