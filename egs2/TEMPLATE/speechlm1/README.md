# ESPnet2 Speech Language Model (SpeechLM) Recipe

## Table of Content

- [ESPnet2 Speech Language Model (SpeechLM) Recipe](#espnet2-speech-language-model-speechlm-recipe)
  - [Table of Content](#table-of-content)
  - [Environment](#environments)
  - [Check List of Building a New Task](#check-list-of-building-a-new-task)
  - [The Concept of Task Templete](#the-concept-of-task-templete)
    - [LM modeling paradigm](#lm-modeling-paradigm)
    - [Define Task as a Sequence Template](#define-task-as-a-sequence-template)
    - [Define a New Task](#define-a-new-task)
    - [Build Model with a Defined Sequence Template](#build-model-with-a-defined-sequence-template)
  - [Recipe Flow](#recipe-flow)
    - [Stage 1: Data Preparation](#stage-1-data-preparation)
    - [Stage 2: Audio Formatting](#stage-2-audio-formatting)
    - [Stage 3-4: Left for Future](#stage-3-4-left-for-future)
    - [Stage 5: Tokenization](#stage-5-tokenization)
    - [Stage 6: Build Joint Vocabulary](#stage-6-build-joint-vocabulary)
    - [Stage 7: Collect Statistics](#stage-7-collect-statistics)
    - [Stage 8: Training](#stage-8-training)
    - [Stage 9: Inference](#stage-9-inference)
    - [Stage 10: Evaluation](#stage-10-evaluation)
    - [Stage 11-12: Model Sharing](#stage-11-model-sharing)
    - [Stage 13: Dataset Sharing](#stage-12-dataset-sharing)
  - [Miscellaneous](#miscellaneous)
    - [Entry Name Rules](#entry-name-rules)
    - [List of Supported Modalities](#list-of-supported-modalities)
    - [List of Supported Type](#list-of-supported-type)
    - [Example Sequence](#example-sequence)
    - [List of Supported Task](#list-of-supported-task)
    - [Supported SpeechLM Model Architecture](#supported-speechlm-model-architecture)
    - [HuggingFace Transformer Implementation and Pre-trained Models](#huggingface-transformer-implementation-and-pre-trained-models)
    - [Build a Evaluation Script](#build-a-evaluation-script)
    - [Data Loading and Preprocessing](#data-loading-and-preprocessing)
  - [Resources](#resources)
  - [FQA](#fqa)

## Environments
First, please install ESPnet following the [Install Instruction](https://espnet.github.io/espnet/installation.html).
  * Recommend to use Pytorch 2.1.0 and above

If you plan to use third-party models from Huggingface, also install:
```
# HuggingFace Tools
pip install huggingface-hub transformers tokenizers datasets

# Flash Attention
pip install flash-attn --no-build-isolation
```

For evaluation purpose, also install VERSA (ESPnet evluation toolkit, under rapid development).
```
git clone https://github.com/ftshijt/speech_evaluation.git
cd speech_evaluation
pip install .
```
VISQOL dependency may have some issue. If you don't need that, comment [this line](#https://github.com/ftshijt/speech_evaluation/blob/50419bda43c27a0c3c484e96214bf5e02dbed089/setup.py#L54) and then install

TODO: Build DeepSpeed environment

## Check List of Building a New Task
We provide a check list for developers who want to work on a new task with ESPnet SpeechLM. Users are highly recommended to read [The Concept of Task Templete](#the-concept-of-task-templete) before working on a new task.
  * Task Template Definition:
    * Check if your task already has a template.
      * All existing task template definitions are in: `<espnet>/espnet2/speechlm/definitions.py`
    * If not, define it following [Define Task as a Sequence Template](#define-task-as-a-sequence-template) and [Define a New Task](#define-a-new-task).
  * Dataset Preparation:
    * Based on task template, prepare the original data files following [Build Model with a Defined Sequence Template](#build-model-with-a-defined-sequence-template) and [Stage 1: Data Preparation](#stage-1-data-preparation).
  * Evaluation Script
    * Make sure what metrics are needed in your task
    * Check [Stage 10: Evaluation](#stage-10-evaluation) and [Build a Evaluation Script](#build-a-evaluation-script)

Ideally, stages like tokenization, data statistics collection, training and inference are all handled automatically. However, it is usually beneficial if you can also inspect the following items:
  * In [Stage 5: Tokenization](#stage-5-tokenization), check if `data.json` is properly built and all files listed in it exist.
  * In [Stage 6: Build Joint Vocabulary](#stage-6-build-joint-vocabulary), check if the joint vocabulary is built as expected.
  * Training and Inference configurations.
  * Many modality-specific configurations. E.g.,
    * Codec model and SSL model choices;
    * BPE model and g2p model choices.
  * Check the inference results, specifically ensure the training and teacher-forced inference have the same results.

Developers are also highly encouraged to adopt our resources, such as codec model, ssl model and pre-trained language models. We also provide an example recipe. See [Resources](#resources).

## The Concept of Task Templete
### LM Modeling Paradigm
Many tasks are to generate a `target` with the given `condition`. In that case, the neural networks usually model the posterior distribution P(`target`|`condition`). E.g.,
  * Automatic Speech Recognition (ASR): P(`text` | `speech`)
  * Speech Enhancement (SE): P(`clean_speech` | `noisy_speech`)
  * Text-to-Speech (TTS): P(`target_speech` | `text, speaker_prompt`)
  * Voice Conversion (VC): P(`target_speech` | `source_speech, reference_speech`)
  * ...

With this very general `condition`-`target` formulation, one can unify these tasks into a language model (LM) paradigm by:
  * Convert both `target` and `condition` into discrete token squences;
  * Splice all discrete token sequences as a whole sequence for language modeling.
  * E.g., a sequence for ASR task (`condition`=speech, `target`=text) may conceptually look like:
    * `<speech_token1> ... <speech_tokenN> <text_token1> ... <text_tokenM>`

The LM is then trained on these spliced sequences, specifically with cross-entropy loss applied to the `target` part of each sequence. During inference, the model can generate the predicted `target` token sequence with the given `condition` sequence, using search algorithms such as greedy search, beam search and sampling. The ultimate system output (audio, text etc.) is recovered by these `target` token sequence.
  * i.e., Input data -> Tokenize -> Input Tokens -> LM Modeling -> Output Tokens -> Detokenize -> Output Data

### Define Task as a Sequence Template
The key philosophy of ESPnet SpeechLM toolkit is to define each task as a sequence template. By doing so, one can quickly extend our code to an unseen task and train an LM with their own data.

Typically, a sequence template consists of several ordered `entries`. Each `entry` represents a specific kind of information used in this task. `Entries` for `condition` will come first then followed by those for `target`.
  * E.g., For ASR task:
    * The first `entry` is speech, and is a `condition` entry;
    * The second `entry` is text, and is a `target` entry.
  * E.g., For TTS task:
    * The first `entry` is phone sequence, and is a `condition` entry;
    * The second `entry` is speaker prompt, and is also a `condition` entry;
    * The third `entry` is speech, and is a `target` entry.

For each `entry`, the user should define three factors: `name`, `modality` and `type`.
  * `name`: each `entry` should have a unique name in that template, so that different entries in the same modality can be distinguished from each other. This is also used as the file name for that `entry` during data preparation stage.
  * `modality`: ESPnet SpeechLM is naturally multi-modal and will have modality-specific operations in many scenarios, for which the modality definition for each `entry` needs to be specified. Primarily, we support text and speech. However, both text and speech can be converted into discrete tokens by multiple methods, and these tokens generated from different tokenization methods will be considered as different modalities. (E.g., speech tokenized by codec and self-supervised model will be considered as in different modalities).
  * `type`: this will define how our dataloader can parse the data for this `entry`. Some data can be simply parsed as plain text, while others may need other parsing methods.

Thus, to define a sequence template for a task is to define `name`, `modality` and `type` for each `entry`. Here is the sequence template for codec-based ASR (codec is a kind of model that convert continuous speech into discrete tokens):
```
# Codec-based ASR Sequence Template
SPEECHLM_TASKS["asr"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec", "kaldi_ark")],
    targets=[("text", "text_bpe", "text")],
)
```
  * The speech entry is defined as the triplet `("wav.scp", "codec", "kaldi_ark")`
    * `name=wav.scp`: the entry is named as `wav.scp`
    * `modality=codec`: this entry is speech and is tokenized by codec model
    * `type=kaldi_ark`: data for this entry should be parsed as a kaldi_ark
  * The text entry is defined as the triplet `("text", "g2p", "text")`
    * `name=text`: the entry is named as `text`
    * `modality=text_bpe`: this entry is text and is tokenized by BPE
    * `type=text`: data for this entry should be parsed as plain text
  * With this template, the dataloader will load the data, convert them into discrete sequence and splice them together for model training / inference. See [Example Sequence](#example-sequence) and [Data Loading and Preprocessing](#data-loading-and-preprocessing) for details.
  * See [List of Supported Task](#list-of-supported-task) for more examples on sequence template definitions

### Define a New Task
All sequence templates are stored in file `<espnet>/espnet2/speechlm/definitions.py`. To add a new task, the user should add a new item in the dict `SPEECHLM_TASKS`:
```
# Codec-based ASR Sequence Template
SPEECHLM_TASKS[<task_name>] = SpeechLMTaskTemplate(
    conditions=[...],
    targets=[...],
)
```
  * First, the users need to figure out what are `condition` and `target` entries in the task.
    * To avoid duplicated definitions, see [List of Supported Tasks](#list-of-supported-task) before you add a new one.
  * Second, for each entry:
    * Assign a `name` for it
      * See [Entry Name Rules](#entry-name-rules)
    * Select the `modality`
      * See [List of Supported Modalities](#list-of-supported-modalities)
    * Select the `type`
      * See [List of Supported Type](#list-of-supported-type)

### Build Model with a Defined Sequence Template
Once the sequence template is defined, users should also prepare the dataset. Ideally, stages of tokenization, training, inference and evaluation are handled automatically by our ESPnet SpeechLM script. See [Recipe Workflow](#recipe-flow) for details.

During the data preparation stage, the user only needs to create a data index file for each `entry` in the task sequence template, with the identical file name of `name` from that entry.
  * E.g., for ASR task, simply prepare two files: `wav.scp` and `text`.
There are multiple examples in each dataset; each example will have an `example_id`. Usually, each index file follows the format of `<example_id> <content>`.
    * For ASR task, index files look like:
```
# wav.scp
example_id1 path-to-wav1
exmaple_id2 path-to-wav2
...

# text
example_id1 transcript1
example_id2 transcript2
...
```
The user will need to prepare data for train / valid / test. Thus, the data folder is of the structure:
```
data
  |--train
  |   |--text
  |   |--wav.scp
  |--valid
  |   |--text
  |   |--wav.scp
  |--test
  |   |--text
  |   |--wav.scp
```

Then, the user can call the `speechlm.sh` for all remained stages, with the provided training and inference configuration files.
  * Remember to specify the `--task` option
  * Also get a name for you dataset using `--data_name`
```
./speechlm.sh \
    --task "asr" \
    --data_name <name-of-your-data> \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --train_set train \
    --valid_set valid \
    --test_sets test
```

## Recipe Flow
ESPnet SpeechLM recipe consists of 12 stages.

  * Before we proceed, we should assume the users already have defined the task in `<espnet>/espnet2/speechlm/definitions.py` (see [Guidance](#define-sequence-template-for-a-new-task)).
  * We will also assume you are in the directory of `<espnet>/egs2/<dataset_name>/speechlm1`.
    * If that directory doesn't exist, build it with `<espnet>/egs2/TEMPLATE/speechlm1/setup.sh`
  * We demonstrate the workflow of ASR task. Users working on other tasks should have the similar procedure.
  * All workflow is in `speechlm.sh`. We create a `run.sh` to call it with the options that are specific to our task and dataset. We shall add `--task` and `--data_name` argument when calling `speechlm.sh`
    * Never call `speechlm.sh` directly. Use `run.sh` to call it install.
```
# run.sh
./speechlm.sh \
  --task asr \
  --data_name <name-of-your-data> \    # e.g., LibriSpeech
  ...
```

### Stage 1: Data Preparation
The data preparation stage is totally customized according to different task and dataset. Users are responsible to make a shell script `local/data.sh` to handle all data preparation process that is specific to your recipe. This script will be called automatically by `speechlm.sh` in this stage.

The data preparation stage is totally flexible to users as long as the final outcome follows the format as described in [Build Model with a Defined Sequence Template](#build-model-with-a-defined-sequence-template). The prepared files should be placed in `./data` folder.

Remember to specify the train / valid / test dataset names to proceed:
```
./speechlm.sh \
  ...
  --train_set "${train_set}" \
  --valid_set "${valid_set}" \
  --test_sets "${test_set}"  \
  ...
```

### Stage 2: Audio Formatting
Many audio files are with different configurations (file format, sampling rate etc.). In this stage, we will standardize all audios for follow-up processing. Based on the sequence template, `entries` that represents audio will all experience this process (i.e., `modality={ssl, codec}`).
Besides audio, the files from other non-audio `entries` will also be inspected so that lines with empty content (i.e., #filed <=1) will be excluded.
  * The results are in `${dump}/audio_raw_${task}_${data_name}`
  * The results of this stage usually takes large space on disk. Users can safely remove it after [tokenization (Stage 5)](#stage-5-tokenization).

### Stage 3-4: Left for Future
### Stage 5: Tokenization
The tokenization stage process the files in `${dumpdir}/audio_raw_${task}_${data_name}` and then generated the results in `${dumpdir}/raw_${task}_${data_name}`. This stage will complete the following three goals:
  * Tokenization
  * Vocabulary Generation
  * Make `data.json`

#### Tokenization
The tokenization process is applied to the index file of each `entry` and is mainly based on the `modality` of it. The tokenization for some modalities is light and can work on-the-fly (e.g., BPE tokenization), while others may have heavy computing and should be processed offline. The online/offline tokenization for different modalities are described in [List of Supported Modalities](#list-of-supported-modalities).

#### Vocabulary Generation
Different `entries` can come from different modalities. Regardless of the online/offline tokenization, a modality-specific vocabulary will be generated and used in [stage 6](#stage-6-build-joint-vocabulary)

#### Make data.json
After all tokenization is done and all modality-specific vocabularies are built, we will build a `data.json` file by script `pyscripts/utils/make_speechlm_json.py`. This will organize all resources and meta-info for this task and this dataset.
  * During training and inference, dataloader can be directly built based on `data.json`.
    * See [Data Loading and Preprocessing](#data-loading-and-preprocessing).
  * Sanity Check: we also exclude the examples that fail to have all `entries` in the provided index files.

```
# An example of data.json:
{
    "task": "asr",
    "vocabularies": [
        "dump/raw_asr_librispeech/test_clean/token_lists/codec_token_list",
        "dump/raw_asr_librispeech/test_clean/token_lists/text_bpe_token_list"
    ],
    "data_files": [
        "dump/raw_asr_librispeech/test_clean/index_files/wav.scp,codec,kaldi_ark",
        "dump/raw_asr_librispeech/test_clean/index_files/text,text_bpe,text",
    ],
    "num_examples": 2220,
    "examples": [
        "1089-134686-0000",
        "1089-134686-0001",
        ...
    ]
}
```

### Stage 6: Build Joint Vocabulary
This stage will merge all vocabularies to form the final vocabulary that we will use for SpeechLM training and infernece. This process relies on the `vocabularies` list in `data.json`. The generated vocabulary has the following layout:
```
[0-255]: reserved for special tokens
  [0-31]: general special tokens, e.g., `<sos/eos>`, `<pad>` etc;
  [32-63]: modality identififer, e.g., `<codec_start/end>` etc;
  [64-127]: task identifier, e.g., `<tts_task>` etc;
  [128-255]: reserved for future usage.
Spliced normal tokens from each modality. E.g., in ASR
  [256-8447]: `codec` tokens
  [8448-13447]: `text_bpe` tokens
```
  * Usually, users will not need to add new special tokens manually.
  * Add your special tokens to interval [0, 31] if you indeed need them.

We still need to record the boundary (starting point) of each modality within the merged vocabulary, which is saved as the `token_bias.json`. This file will help to conduct modality-specific operations during the training and inference.
```
# token_bias.json
{
    "codec": 256,
    "text_bpe": 8448
}
```

### Stage 7: Collect Statistics
This stage will collect the statistics of each example, typically the sequence length. The length statistics will be later used in batchfying the training examples.
The statistics will be kept in the folder `exp/speechlm_stats_<task>_<data_name>`. E.g., `exp/speechlm_stats_tts_librispeech`.
```
# exp/speechlm_stats_asr_librispeech/train/dec_seq_lengths
asr_100-121669-0001 459
asr_100-121669-0002 891
...
```
  * Along this way, we will also shared the `data.json` file by ${ngpu}. See [Sharded Dataset](#sharded-dataset)
  * A task prefix `asr_` is added to each `<example_id>`. See [Multi-Task Training and Multi-Task Dataset](#multi-task-training-and-multi-task-dataset)

### Stage 8: Training
The training process is to train the LM using the spliced token sequences and compute the cross-entropy loss over the `target entry` segments.
  * The top-level model is in `espnet2/speechlm/espnet_model.py`, which is a warpper for the real SpeechLM implementation `corelm`.
    * `corelm` refer to a SpeechLM architecture and are in `<espnet>/wse3espnet2/speechlm/corelm`
  * Regardless what `corelm` you choose, its input interface for forward are the same:
    * `dec_seq`: decoder sequence, of size `[B, T, N]`.
    * `dec_seq_lengths`: the effective length of decoder sequence, of size `[B]`.
    * `enc_seq`: encoder sequence, of size `[B, T, N]`.
    * `enc_seq_lengths`: the effective length of encoder sequence, of size `[B]`.
    * `prefix_len`: the length of `condition` part of each sequence, of size `[B]`.
      * B: batch size
      * T: length in time-axis, i.e., number of frames
      * N: number of codes per frame
    * Unlike standard LMs that work on input format `[B, T]`, all SpeechLM will assume there are `N` tokens for each modeling units. See [List of Supported Modality](#list-of-supported-modalities).

  * Optimizer, scheduler, scaler, checkpoint saving / resume etc. follow standard ESPnet2
  * Also see:
    * [Supported SpeechLM Model Architecture](#supported-speechlm-model-architecture)
    * [HuggingFace Transformer Implementation and Pre-trained Models](#huggingface-transformer-implementation-and-pre-trained-models)
    * [Data Loading and Preprocessing](#data-loading-and-preprocessing)
    * [Distributed Training, Automatic Precision Training and Numerical Stability](#distributed-training-automatic-precision-training-and-numerical-stability)

### Stage 9: Inference
During the inference, the basic idea is to generate `target` based on `condition`. For SpeechLM, this is to generate `target` sequence based on the `condition` sequence.
  * For each `corelm` architecture, there is a `inference` implementation.
  * For the search algorithms, we currently support:
    * Top-k sampling
    * Top-p sampling
    * Greedy Search

After SpeechLM inference, the `target` tokens will experience detokenization process so that the ultimate system output is obtained (e.g., audio, text). The detokenization process depends on the `modality` of `target entry`. See [List of Supported Modalities](#list-of-supported-modalities).

### Stage 11-12: Model Sharing
Similar to other ESPnet modules, SpeechLM models can also be released by HuggingFace. See [Contributing.md](https://github.com/espnet/espnet/blob/master/CONTRIBUTING.md#132-espnet2-recipes)

### Stage 13: Dataset Sharing
Besides sharing the model, developers can also share their data, specifically the tokenized data described by `data.json`. As described in [Multi-Task Training and Multi-Task Dataset](#multi-task-training-and-multi-task-dataset), a team can easily build a data mixture for multi-task training once its members share their data for each task and each dataset.

To share the dataset, make sure you already have the huggingface account and then login:
  * `huggingface-cli login`

Then create a dataset repository with your `data_name` as:
  * `huggingface-cli repo create -y ${data_name} --type dataset`

Upload your data to dataset. Typically we would like to requeest you to upload the ${data_feats} folder that contains all tokenized results:
  * `huggingface-cli upload --repo-type dataset ${data_name} ${data_feats} ${data_feats}`
  * `${data_feats}` folder is in the format of `data_feats="${dumpdir}/raw_${task}_${data_name}"`. E.g., `dump/raw_tts_libritts`


## Miscellaneous
### Entry Name Rules
The names of the entries are usually flexible, but still with some constraints:
  * If there is an entry of modality `spk` (for speaker prompt), there should be another entry called `wav.scp` as the speaker prompt will be selected from the `wav.scp` on-the-fly.

Also, for consistency, it is highly recommended to follow the [kaldi-style file format](#https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE#about-kaldi-style-data-directory).

### List of Supported Modalities
| Data Description            | Code  | Shape  | Offline Tokenization | Detokenization |
|-----------------------------|-------|--------|----------------------|----------------|
| Audio, tokenized by Codec models | `codec` | [T, N] | Yes | Yes                 |
| Speech, tokenized by SSL models  | `ssl`   | [T, 1] | Yes | No            |
| Text, tokenized by BPE model | `text_bpe` | [T, 1] | No | Yes |
| Text, tokenized by G2P tool | `g2p` | [T, 1] | No | No |
| Speaker-ID, as a utterance-to-speaker mapping | `spk` | [T, N] | No | No |
| Classification label | `class` | [T, 1] | No | Yes |
Hint:
  * `T` is the sequence length; `N` is the number of codes for each frames in codec models.
  * Initially, `spk` is in the textual format of `example_id` `speaker-id`. During data loading, a speaker-to-utterance mapping is built, base on which the speaker prompt is randomly selected from another utterance of the same speaker, which is still represented by `codec` tokens and is of size `[T, N]`.

### List of Supported Type
Most users should be confortable to stay with `type` of `text` and `kaldi_ark`, both of which have the format `example_id` `content` in each line of it.
  * If the file is saved as kaldi_ark, use `kaldi_ark`
    * Usually, tokenized speech (`modality={codec, ssl}`) is in this format.
    * Each line will look like: `example_id <path-to-ark>:<index_number>`. E.g., foo.ark:1234
  * If the file can be effectively parsed as plain text, use `text`.
    * usualy, data with `modality={text_bpe, g2p, spk, class}` is in this foramt.
For advanced users, ESPnet dataset will support more `type` options. See `DATA_TYPES` in `<espnet>/espnet2/train/dataset.py` for the full list.

### Example Sequence
When using speech codec models to represent audio/speech, it will use `N` tokens for each frame, which is not compatible with standard LM. Thus, for any modality other than codec, we repeat it by `N` times before splicing all sequence entries together. The final sequence will be in the shape of `[T, N]`.

Besides the tokens from each modality, we additionally add some special tokens for flow control and other logics, such as
  * Start/End of Sentence `<sos/eos>`
  * Task Identifier, e.g., `<asr_task>`
  * Modality Identifier, e.g., `<codec_start/end>`

So that, a complete sequence for ASR task (`N=3`) can look like:
```
[
 [<sos/eos>,            <sos/eos>,            <sos/eos>           ],
 [<asr_task>,           <asr_task>,           <asr_task>          ],
 [<codec_start/end>,    <codec_start/end>,    <codec_start/end>   ],
 [codec_layer1_frame1,  codec_layer2_frame1,  codec_layer3_frame1 ],
 [codec_layer1_frame2,  codec_layer2_frame2,  codec_layer3_frame2 ],
 ...
 [<text_bpe_start/end>, <text_bpe_start/end>, <text_bpe_start/end>],
 [text_bpe1,            text_bpe1,            text_bpe1           ],
 [text_bpe2,            text_bpe2,            text_bpe2           ],
 ...
 [<sos/eos>,            <sos/eos>,            <sos/eos>           ],
]
```

### List of Supported Task
Below is a list of task we currently support.
  * Each triplet stands for an `entry` and consists of `name`, `modality` and `type`.
  * The real task list in use is in `<espnet>/espnet2/speechlm/definitions.py`.

| Task Name | Task Full Name                  | Condition Triplet                       | Target Triplet                  |
|-----------|---------------------------------|----------------------------------|--------------------------|
| `textlm`    | Text Language Model | - | (text,text_bpe,text)
| `audiolm` | Audio Language Model | - | (wav.scp,codec,kaldi_ark)
| `asr`       | Automatic Speech Recognition    | (wav.scp, codec, kaldi_ark)      | (text, text_bpe, text)   |
| `mt` | Machine Translation | (src_text,text_bpe,text) | (text,text_bpe,text)
`tts` | Text-to-Speech | (text,g2p,text), (utt2spk,spk,text) | (wav.scp,codec,kaldi_ark)
`se` | Speech Enhancement | (noisy.scp,codec,kaldi_ark) | (wav.scp,codec,kaldi_ark)
`st` | Speech Translation<br>(with source language) | (wav.scp,codec,kaldi_ark) | (src_text,text_bpe,text), (text,text_bpe,text)
...

### Supported SpeechLM Model Architecture
We currently support the following architectures that are specifically designed for SpeechLM
  * [Flatten (Standard LM)](https://arxiv.org/abs/2306.05284)
  * [Vall-E](https://arxiv.org/abs/2301.02111)
  * [Parallel](https://arxiv.org/abs/2306.05284)
  * [Delay](https://arxiv.org/abs/2306.05284)
  * [MutiScale Transformer](https://arxiv.org/abs/2310.00704)

Note: Vall-E cannot be used when data in `codec` modality doesn't exist or `N=1`. This is because there are no codes for Vall-E non-auto-regressive modeling.

### HuggingFace Transformer Implementation and Pre-trained Models
We provide a unified interface for ESPnet built-in Transformer and HuggingFace Transformer models.
  * Check the warpper module in `<espnet>/epsnet2/speechlm/module/transformer.py`
  * When `hf_model_tag` is not set, ESPnet builtin Transformer is used;
  * When `hf_model_tag` is set, the corresponidng HuggingFace Transformer model and its weight is loaded, the `text_bpe` part of the embedding table and lm_head will be overrided.
    * If use Huggingface model, make sure you set the consistent BPE model. Check the `--bpemode` and `--bpemodel` argument in `speechlm.sh`
  * The unified Transformer interface only takes care of stacked Transformer layers and positional embeddings; the token embedding and hidden-to-logits process is kept in each `corelm` implementation.

### Distributed Training, Automatic Precision Training and Numerical Stability
  * For small model training with more than one GPU, distributed data parallel (DDP) will be the most efficient option. simply set `--ngpu` to `speechlm.sh` should be ok.
  * For large model training (typically > 1B), Pytorch builtin Fully-Sharded Data Parallel (FSDP) is currently supported. To use it, set (1) `--use_fsdp` in your traning config and (2) ensure your Transformer layer class is in `layer_cls` list of `<espnet>/espnet2/speechlm/espnet.model.py`.
  * Automatic Mixed Precision (AMP) Training is recommended to use (with training config argument `--use_amp true`) but will cause numerical instability sometimes, especially with `fp16` data type. If the users work with Ampere GPUs and above, `bf16` data type is used by default and will be more stable. For ESPnet built-in Transformer, we recommend to use `qk_norm` option to stabilize the training.
  * We plan to support DeepSpeed in the future.

### Build a Evaluation Script
Advance users who extend ESPnet SpeechLM to a new task usually need to create a evaluation script `scripts/utils/speechlm_eval/eval_<task>.sh`. This script will compute multiple metrics based on the generated content and then output a evaluation summary.
  * See `scripts/utils/speechlm_eval/eval_tts.sh` as an example

The evaluation script would contain two parts:
  * Compute the result for each metric
    * For each metric, it should generate a result file `utt_result.json`.
      * Each line should contain the example_id (`key`) and metric name
      * Optionally, specify a feature `weight` for this example. E.g., word count when computing WER for ASR.
      * E.g., `{"key": "tts_id00001-001_sample0", "wer": 9, "weight": 10}`

  * Summarize all `utt_result.json` and generate the final statistics.
    * we provide `pyscripts/utils/result_summary.py` for this purpose.

Developers are highly recommend to use [VERSA](https://github.com/ftshijt/speech_evaluation) for evaluation purpose. Developrs can simply compute a metric by providing a simple json config. Currently, we support:
  * MCD
  * F0-RMSE
  * F0-CORR
  * STOI
  * PESQ
  * CI-SDR
  * D-BLEU
  * D-Distance
  * S-BERT
  * VISQOL
  * UTMOS
  * PLCMOS
  * DNSMOS
  * WER
  * SPK-S

### Data Loading and Preprocessing
The core code for data load and preprocessing are in the following objects:
  * `ESPnetDataset`, `EspnetSpeechLMDataset`, `ESPnetMultiTaskDataset` in `<espnent>/espnet2/train/dataset.py`, for data loading logics
  * `SpeechLMPreprocessor` in `<espnent>/espnet2/train/preprocessor.py`, for data preprocessing logics.

#### ESPnet Dataset
For each given `data.json`, a `EspnetSpeechLMDataset` object can be built automatically, which is largely inherited by `ESPnetDataset` object. We will firstly introduce the `ESPnetDataset` module.

Remember in the [data.json](#make-datajson) section, we have the `all_files` list in the given `data.json`:
```
"data_files": [
    "dump/raw_asr_librispeech/test_clean/index_files/wav.scp,codec,kaldi_ark",
    "dump/raw_asr_librispeech/test_clean/index_files/text,text_bpe,text",
]
```
A dict called `loader_dict` is built with the lines in `data_files`. For each line, we load the index file with `type` reader and then save it with the key of `name`
```
# In ESPnet Dataset object
-- self.loader_dict
      |-- wav.scp : dict-like kaldi_ark reader
      |-- text    : dict-like text reader
```
Then, a example dict can be obtained with a given `example_id`
```
# example dict
data = {
  "wav.scp": self.loader_dict["wav.scp"][example_id]
  "text": self.loader_dict["text"][example_id]
}
```
Due to the different `type`s of these loaders, there could be many kinds of data in this example dict, such as numpy array, text etc. The example dict is then processed by the preprocessor and then output the composed token sequence for model training.

#### Preprocessing
For each example dict, the preprocessor will compose it into a training sequence. This process is mainly based on the sequence template of the task.
For each entry in the sequence template:
  * we first fetch the value in the example_dict by the `name` of that entry.
    * `vallue = data[name]`
  * And then do some modality-specific operation to this value:
    * `value = self.modality_specific_processing(value, modality)`
    * There would be many different behaviors for different modalities. E.g.,
      * Resize and expand the discrete tokens to format of `[T, N]`
      * On-the-fly BPE / G2P tokenization etc.
      * ...
  * Fianlly, splice all these `value`s together following the original sequence template definition
    * Will also add some special tokens
    * For ASR, as in [example sequence](#example-sequence)

Dataset-Level Preprocessing: The input to the preprocessor is the example dict, which only contains the data items of that exmaple. However, some on-the-fly preprocessing may leverage other examples, which raises the needs of dataset-level preprocessing.

One of the additional feature for `EspnetSpeechLMDataset` is to support this dataset-level operation. An example is to support the speaker prompt (`modality=spk`). When initialize the dataset, we first initialize the `spk2utt` dictionalry based on the `utt2spk` input file. When queried by an example_id, we find the speaker of it and then randomly find another example from that speaker to get the speaker prompt.

#### Multi-Task Training and Multi-Task Dataset
Conceptually, a multi-task SpeechLM can be built by training it with example sequences from various tasks. E.g., having sequences for ASR, TTS, etc. in the same training batch.

ESPnet SpeechLM can easily support multi-task training by simply input multiple `data.json` files during training. The users can use the stage 1-5 of `speechlm.sh` to prepare the `data.json` for each of the single task and different dataset. From stage 6 and later, instead of specifying `train_set`, `valid_set` and `test_set` as before, the users can directly work with a list of `data.json` files. E.g.,
```
train_jsons=" \
  dump/asr/train/data.json \
  dump/tts/train/data.json \
  dump/se/train/data.json
  ...
"
valid_jsons=" \
  dump/asr/valid/data.json \
  dump/tts/valid/data.json \
  dump/se/valid/data.json
  ...
"
./speechlm.sh \
  --stage 6 \
  --train_jsons ${train_jsons} \
  --valid_jsons ${valid_jsons} \
  ...
```

Internlly, each input `data.json` will be used to build a specific `EspnetSpeechLMDataset`; all these `EspnetSpeechLMDataset` are coordiated with the `ESPnetMultiTaskDataset` object.
```
ESPnetMultiTaskDataset
  | -- EspnetSpeechLMDataset (ASR - LibriTTS)
  | -- EspnetSpeechLMDataset (ASR - GigaSpeech)
  | -- EspnetSpeechLMDataset (TTS - LibriTTS)
  | -- EspnetSpeechLMDataset (SE - Chime)
  ...
```
Then the example_list of each EspnetSpeechLMDataset will be aggregated to a whole list in ESPnetMultiTaskDataset. These dataset will be accessed in a fused manner based on the batching results.

Note, we always use the `ESPnetMultiTaskDataset` as a unified interface even though this is only one `data.json` for train or valid.

#### Sharded Dataset
When training with massive data, storing the whole dataset in each GPU process is expensive. So that, in [stage 7](#stage-7-collect-statistics), we shard each input `data.json` based on the number of GPUs.

## Resources
### Tokenization Models
`Codec`: As of Aug 10, 2024, we encourage the developers to use our open-sourced codec-model from Jiatong:
  * https://huggingface.co/espnet/owsmdata_soundstream_16k_200epoch
  * In `speechlm.sh`, use it with argument `--codec_choice ESPnet --codec_hf_model_tag espnet/owsmdata_soundstream_16k_200epoch`

`SSL`: As of Aug 10, 2024, we encourage the developers to use our open-sourced codec-model from William:
  * https://huggingface.co/datasets/JinchuanTian/speechlm_ssl_xues
  * First, download the model by `huggingface-cli download --repo-type dataset --local-dir . JinchuanTian/speechlm_ssl_xues`

### Pretrained Models:
As of Sep 18, we provide two pre-trained models. These models are trained on 160khrs of English corpus, of size 300-400M parameters. People who work on speech understanding can use ASR pre-trained model; people who work on speech generation can use TTS pre-trained model.

`ASR`: https://huggingface.co/datasets/espnet/espnet_speechlm_pretrained_asr
  * download the model: `cd <espnet>/egs2/<recipe_name>/speechlm1; huggingface-cli download --repo-type dataset --local-dir . espnet/espnet_speechlm_pretrained_asr`
  * Use the training config file: `<espnet>/egs2/librispeech/speechlm1/conf/train_delay_asr.yaml`. You should keep the model configuration unchanged, but feel free to revise other configs.
  * Use the prepared token list folder. In `speechlm.sh`, use `--token_list_dir data/token_list/asr_vocab`. The folder is together downloaded with the model.

`TTS`: https://huggingface.co/datasets/espnet/espnet_speechlm_pretrained_tts
  * download the model: `cd <espnet>/egs2/<recipe_name>/speechlm1; huggingface-cli download --repo-type dataset --local-dir . espnet/espnet_speechlm_pretrained_tts`
  * Use the training config file: `<espnet>/egs2/librispeech/speechlm1/conf/train_delay_tts.yaml`. You should keep the model configuration unchanged, but feel free to revise other configs.
  * Use the prepared token list folder. In `speechlm.sh`, use `--token_list_dir data/token_list/tts_vocab`. The folder is together downloaded with the model.

### Recipes：
  * `ASR` fine-tuning the pre-trained model: `<espnet>/egs2/librispeech/speechlm1/run_asr.sh`
  * `TTS` fine-tuning the pre-trained model: `<espnet>/egs2/librispeech/speechlm1/run_tts.sh`

## FQA
