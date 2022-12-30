---
title: 'Software Design and User Interface of ESPnet-SE++: Speech Enhancement for Robust Speech Processing'
tags:
  - Python
  - ESPnet
  - speech processing
  - speech enhancement
authors:
  - name: Yen-Ju Lu
    orcid: 0000-0001-8400-4188
    equal-contrib: true
    affiliation: 1
  - name: Xuankai Chang
    orcid: 0000-0002-5221-5412
    equal-contrib: true
    affiliation: 2
  - name: Chenda Li
    orcid: 0000-0003-0299-9914
    affiliation: 3
  - name: Wangyou Zhang
    orcid: 0000-0003-4500-3515
    affiliation: 3
  - name: Samuele Cornell
    orcid: 0000-0002-5358-1844
    affiliation: "2, 4"
  - name: Zhaoheng Ni
    affiliation: 5
  - name: Yoshiki Masuyama
    affiliation: "2, 6"
  - name: Brian Yan
    affiliation: 2
  - name: Robin Scheibler
    orcid: 0000-0002-5205-8365
    affiliation: 7
  - name: Zhong-Qiu Wang
    orcid: 0000-0002-4204-9430
    affiliation: 2
  - name: Yu Tsao
    orcid: 0000-0001-6956-0418
    affiliation: 8
  - name: Yanmin Qian
    orcid: 0000-0002-0314-3790
    affiliation: 3
  - name: Shinji Watanabe
    corresponding: true
    orcid: 0000-0002-5970-8631
    affiliation: 2
affiliations:
  - name: Johns Hopkins University, USA
    index: 1
  - name: Carnegie Mellon University, USA
    index: 2
  - name: Shanghai Jiao Tong University, Shanghai
    index: 3
  - name: Universita\` Politecnica delle Marche, Italy
    index: 4
  - name: Meta AI, USA 
    index: 5
  - name: Tokyo Metropolitan University, Japan
    index: 6
  - name: LINE Corporation, Japan
    index: 7
  - name: Academia Sinica, Taipei
    index: 8
date: 22 August 2022
bibliography: paper.bib

---

![](https://i.imgur.com/mV3ukCX.png)

# Summary
This paper presents the software design and user interface of ESPnet-SE++, a new speech separation and enhancement (SSE) module of the ESPnet toolkit. 
ESPnet-SE++ significantly expands the functionality of ESPnet-SE [@Li:2021] with several new models, loss functions, and training recipes [@Lu:2022]. Crucially, it features a new, redesigned interface, which allows for a flexible combination of SSE front-ends with many downstream tasks, including automatic speech recognition (ASR), speaker diarization (SD), speech translation (ST), and spoken language understanding (SLU).

# Statement of need

[ESPnet](https://github.com/espnet/espnet) is an open-source toolkit for speech processing, including several ASR, text-to-speech (TTS) [@Hayashi:2020], ST [@Inaguma:2020], machine translation (MT), SLU [@Arora:2022], and SSE recipes [@Watanabe:2018]. Compared with other open-source SSE toolkits, such as Nussl [@Manilow:2018], Onssen [@Ni:2019], Asteroid [@Pariente:2020], and SpeechBrain [@Ravanelli:2021], the modularized design in ESPnet-SE++ allows for the joint training of SSE modules with other tasks. Currently, ESPnet-SE++ supports 20 SSE recipes with 24 different enhancement/separation models. 


# ESPnet-SE++ Recipes and Software Structure
## ESPNet-SE++ Recipes for SSE and Joint-Task 
![](https://i.imgur.com/zKu612c.png)

For each task, ESPnet-SE++, following the ESPnet2 style, provides common scripts which are carefully designed to work out-of-the-box with a wide variety of corpora. Under the `TEMPLATE` folder, the common scripts `enh1/enh.sh` and `enh_asr1/enh_asr.sh` are shared for all the SSE and joint-task recipes. 

### Common Scripts
`enh.sh` contains 13 stages, and the details for the scripts can be found in [TEMPLATE/enh1/README.md](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/enh1/README.md). 

![](https://i.imgur.com/0rGcwmw.png)

`enh_asr.sh` contains 17 stages and `enh_diar.sh` and `enh_st.sh` are similar to it:

![](https://i.imgur.com/WfB0yVM.png)

### Training Configuration
#### SSE Task Training Configuration
An example of an enhancement task for the CHiME-4 `enh1`  recipe is configured as [`conf/tuning/train_enh_dprnn_tasnet.yaml`](https://github.com/espnet/espnet/blob/master/egs2/chime4/enh1/conf/tuning/train_enh_dprnn_tasnet.yaml). Part of this configuration is: 

 ![](https://i.imgur.com/dsEy0gJ.png)

#### Joint-Task Training Configuration
An example of joint-task training configuration is the CHiME-4 `enh_asr1` recipe, configured as [`conf/tuning/train_enh_asr_convtasnet.yaml`](https://github.com/espnet/espnet/blob/master/egs2/chime4/enh_asr1/conf/tuning/train_enh_asr_convtasnet_si_snr_fbank_transformer_lr2e-3_accum2_warmup20k_specaug.yaml). This joint-task includes a front-end enhancmenet model and a back-end ASR model: 
![](https://i.imgur.com/kTapPT5.png)
![](https://i.imgur.com/uVWW7ft.png)


## ESPNet-SE++ Software Structure for SSE Task
![](https://i.imgur.com/W50IuzE.png)


###  Unified Modeling Language Diagram for ESPNet-SE++ Enhancement-Only Task
![](https://i.imgur.com/YPUERjy.png)


### SSE Executable Code `bin/*`
#### bin/enh_train.py
 As the main interface for the SSE training stage of `enh.sh`, `enh_train.py` takes the training parameters and model configurations from the arguments and calls
 
	EnhancementTask.main(...) 

to build an SSE object `ESPnetEnhancementModel` for training the SSE model according to the model configuration.

#### bin/enh_inference.py
The `inference` function in `enh_inference.py` creates a

	class SeparateSpeech
    
object with the data-iterator for testing and validation. During its initialization, the class builds an SSE object `ESPnetEnhancementModel` based on a pair of configuration and a pre-trained SSE model.

#### bin/enh_scoring.py
	def scoring(..., ref_scp, inf_scp, ...)
The SSE scoring functions calculates several popular objective scores such as SI-SDR [@le:2019], STOI [@Taal:2011], SDR and PESQ [@Rix:2001], based on the reference signal and processed speech pairs.

### SSE Control Class `tasks/enh.py`

	class EnhancementTask(AbsTask)
`EnhancementTask` is a control class which is designed for SSE task, containing class methods for building and training an SSE model. Class method  `build_model` creates and returns an SSE object `ESPnetEnhancementModel`.

### SSE Modules `enh/espnet_model.py`

	class ESPnetEnhancementModel(AbsESPnetModel)
    
`ESPnetEnhancementModel` is the base class for any ESPnet-SE++ SSE model. Since it inherits the same abstract base class `AbsESPnetModel`, it is well-aligned with other tasks such as ASR, TTS, ST, and SLU, bringing the benefits of cross-tasks combination. 

	 def  forward(self, speech_mix, speech_ref, ...)

The `forward` function of `ESPnetEnhancementModel`  follows the general design in the ESPnet single-task modules, which processes speech and only returns losses for the trainer to update the model. 

	 def  forward_enhance(self, speech_mix, ...)
	 def  forward_loss(self, speech_pre, speech_ref, ...)

For more flexible combinations, the `forward_enhance` function returns the enhanced speech, and the `forward_loss` function returns the loss. The joint-training methods take the enhanced speech as the input for the downstream task and the SSE loss as a part of the joint-training loss.

## ESPNet-SE++ Software Structure for Joint-Task 
![](https://i.imgur.com/BPgf1b5.png)

### Unified Modeling Language Diagram for ESPNet-SE++ Joint-Task
![](https://i.imgur.com/qXxjwR5.png)


### Joint-Task Executable Code `bin/*`
#### bin/enh_s2t_train.py
Similarly to the interface of SSE training code `enh_train.py`, `enh_s2t_train.py` takes the training and modular parameters from the scripts, and calls  

	tasks.enh_s2t.EnhS2TTask.main(...) 

to build a joint-task object for training the joint-model based on a configuration with both SSE and s2t models setting with or without pre-trained checkpoints.


#### bin/asr_inference.py, bin/diar_inference.py, and bin/st_inference.py

The `inference` function in `asr_inference.py`, `diar_inference.py`, and `st_inference.py` builds and call a 

	class Speech2Text
    class DiarizeSpeech
object with the data-iterator for testing and validation.  During their initialization, the classes build a joint-task object `ESPnetEnhS2TModel` with pre-trained joint-task models and configurations. 

### Joint-task Control Class `tasks/enh_s2t.py`

	class EnhS2TTask(AbsTask)
    
`class EnhS2TTask` is designed for joint-task model. The subtask models are created and sent into the `ESPnetEnhS2TModel` to create a joint-task object. 


### Joint-Task Modules `enh/espnet_enh_s2t_model.py`
	class ESPnetEnhS2TModel(AbsESPnetModel)

The `ESPnetEnhS2TModel` takes a front-end `enh_model`, and a back-end `s2t_model` (such as ASR, SLU, ST, and SD models) as inputs to build a joint-model.

	def __init__(
	    self,
	    enh_model: ESPnetEnhancementModel,
	    s2t_model: Union[ESPnetASRModel, ESPnetSTModel, ESPnetDiarizationModel],
	    ...
	):
	
The `forward` function of the class follows the general design in ESPnet2:
	 
	 def  forward(self, speech_mix, speech_ref, ...)

which processes speech and only returns losses for `Trainer` to update the model. 


# ESPnet-SE++ User Interface

## Building a New Recipe from Scratch
Since ESPnet2 provides common scripts such as `enh.sh` and `enh_asr.sh` for each task, users only need to create `local/data.sh`  for the data preparation of a new corpus.  The generated data follows the Kaldi-style structure:
![](https://i.imgur.com/aSW6a2M.png)

The detailed instructions for data preparation and building new recipes in espnet2 are described in the following link:

https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE


##  Inference with Pre-trained Models
Pretrained models from ESPnet are provided on HuggingFace and Zenodo. Users can download and infer with the models.`model_name` in the following section should be `huggingface_id` or one of the tags in the [table.csv](https://github.com/espnet/espnet_model_zoo/blob/master/espnet_model_zoo/table.csv) in [espnet_model_zoo](https://github.com/espnet/espnet_model_zoo) . Users can also directly provide a Zenodo URL or a HuggingFace URL.


### Inference API
The inference functions are from the `enh_inference` and `enh_asr_inference` in the executable code `bin/`

    from espnet2.bin.enh_inference import SeparateSpeech
    from espnet2.bin.enh_asr_inference import Speech2Text

Calling `SeparateSpeech` and `Speech2Text` with unprocessed audios returns the separated speech and their recognition results. 

#### SSE 
![](https://i.imgur.com/skZ8uDP.png)
#### Joint-Task
![](https://i.imgur.com/hrj0hJq.png)

The details for downloading models and inference are described in the following link: https://github.com/espnet/espnet_model_zoo


# Demonstrations
The demonstrations of ESPnet-SE can be found in the following google colab links:

- [ESPnet SSE Demonstration: CHiME-4 and WSJ0-2mix](https://colab.research.google.com/drive/1fjRJCh96SoYLZPRxsjF9VDv4Q2VoIckI?usp=sharing) 
- [ESPnet-SE++ Joint-Task Demonstration: L3DAS22 Challenge and SLURP-Spatialized](https://colab.research.google.com/drive/1hAR5hp8i0cBIMeku8LbGXseBBaF2gEyO#scrollTo=0kIjHfagi4T1)


# Development plan
The development plan of the ESPnet-SE++ can be found in https://github.com/espnet/espnet/issues/2200. In addition, we would explore the combinations with other front-end tasks, such as using ASR as a front-end model and TTS as a back-end model for speech-to-speech conversion, making the combination more flexible. 

# Conclusions
In this paper, we introduce the software structure and the user interface of ESPnet-SE++, including the SSE task and joint-task models. ESPnet-SE++ provides general recipes for training models on different corpus and a simple way for adding new recipes. The joint-task implementation further shows that the modularized design improves the flexibility of ESPnet.

# Acknowledgement
This work used the Extreme Science and Engineering Discovery Environment (XSEDE) [@Towns:2014], which is supported by NSF grant number ACI-1548562. Specifically, it used the Bridges system [@Nystrom:2015], which is supported by NSF award number ACI-1445606, at the Pittsburgh Supercomputing Center (PSC).



# References
