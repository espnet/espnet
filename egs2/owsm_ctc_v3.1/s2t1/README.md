# OWSM-CTC v3.1

[OWSM-CTC](https://aclanthology.org/2024.acl-long.549/) is an encoder-only speech foundation model based on hierarchical multi-task self-conditioned CTC.
This version is trained on 180k hours of public audio data for multilingual speech recognition, any-to-any speech translation, and language identification, which follows the design of the project, [Open Whisper-style Speech Model (OWSM)](https://arxiv.org/abs/2401.16658).

## Data Preparation

The training data follows the same format as the encoder-decoder OWSM v3.1, except that timestamps are removed from the `text` file. Please first follow the `egs2/owsm_v3.1/s2t1` recipe to prepare OWSM data, and then convert `text` into the new format by running `python local/convert_owsm_data.py` (the path to the BPE tokenizer needs to be modified to your path).

### OWSM-CTC Data Format

The prepared data directory contains the following files:

```
dump/raw/train
├── feats_type
├── spk2utt
├── text
├── text.ctc
├── text.prev
├── utt2spk
├── wav.scp
```

`feats_type` has a single line of text, which should be automatically generated in the data preparation stage:
```
raw
```

`spk2utt` and `utt2spk` have the same meaning as the standard Kaldi recipes (see `asr1` recipe for example). Typically, the speaker information is not utilized. Hence, each utterance has a unique speaker ID which is simply its utterance ID.

`wav.scp` also follows the standard Kaldi format.

`text` contains the multitask reference (ASR or ST) with language and task tokens but without timestamps:

```
AIDATATANG_200ZH_T0055G0013S0001_000000000_000003561_zho_asr <zho><asr> 今天什么日子
...
GigaST_YOU0000009624_002208970_002218840_en_st_zh <eng><st_zho> 大会结束后,我们要求有兴趣进一步参与我们项目或进一步参与气候教育的学生站出来,
...
MLS_en_sikhreligion6_22_macauliffe_64kb_003555300_003571720_en_asr <eng><asr> it farid considered that faqiri or holiness consisted in four things namely to be blind to the faults of muhammadans to be deaf to slander to be dumb when evil speaking is suggested and to be lame when there is a desire to visit evil places
...
```

`text.ctc` contains the pure ASR reference:

```
AIDATATANG_200ZH_T0055G0013S0001_000000000_000003561_zho_asr 今天什么日子
...
CoVoST2_147d94ad8405722d5930a859295bfac7b925ccd40c587334d34f3ebd2668a70242240866e93907398f10b7f2265a4ddb82b5355eb21fe37993d04a69900df388-common_voice_en_19741894_000000000_000006270_en_st_ca He appointed military officers to most leading government positions.
...
```

`text.prev` contains the previous sentence that will be used as an additional prompt. If a sample does not have a prompt, then `<na>` is used.

```
AIDATATANG_200ZH_T0055G0013S0001_000000000_000003561_zho_asr <na>
...
GigaST_YOU0000009624_002208970_002218840_en_st_zh 与员工和同事一起，这将有助于为事物创造空间，帮助为我们创造空间，一些掩护，尝试新事物，
...
```

## Pre-trained Model

Pre-trained models are available on Hugging Face:
- https://huggingface.co/espnet/owsm_ctc_v3.1_1B
- https://huggingface.co/espnet/owsm_ctc_v3.2_ft_1B

The v3.1 model is trained with this config: [conf/train_s2t_multitask-ctc_ebf27_conv2d8_size1024.yaml](conf/train_s2t_multitask-ctc_ebf27_conv2d8_size1024.yaml)


### Example script for batched inference

`Speech2TextGreedySearch` now provides a unified batched inference method `batch_decode`. It performs CTC greedy decoding for a batch of short-form or long-form audios. If an audio is shorter than 30s, it will be padded to 30s; otherwise it will be split into overlapped segments (same as the "long-form ASR/ST" method below).

```python
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

s2t = Speech2TextGreedySearch.from_pretrained(
    "espnet/owsm_ctc_v3.1_1B",
    device="cuda",
    use_flash_attn=False,   # set to True for better efficiency if flash attn is installed and dtype is float16 or bfloat16
    lang_sym='<eng>',
    task_sym='<asr>',
)

res = s2t.batch_decode(
    "audio.wav",    # a single audio (path or 1-D array/tensor) as input
    batch_size=16,
    context_len_in_secs=4,
)   # res is a single str, i.e., the predicted text without special tokens

res = s2t.batch_decode(
    ["audio1.wav", "audio2.wav", "audio3.wav"], # a list of audios as input
    batch_size=16,
    context_len_in_secs=4,
)   # res is a list of str

# Please check the code of `batch_decode` for all supported inputs
```

### Example script for short-form ASR/ST/LID

Our models are trained on 16kHz audio with a fixed duration of 30s. When using the pre-trained model, please ensure the input speech is 16kHz and pad or truncate it to 30s.

```python
import librosa
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

s2t = Speech2TextGreedySearch.from_pretrained(
    "espnet/owsm_ctc_v3.1_1B",
    device="cuda",
    generate_interctc_outputs=False,
    lang_sym='<eng>',
    task_sym='<asr>',
)

# NOTE: OWSM-CTC is trained on 16kHz audio with a fixed 30s duration. Please ensure your input has the correct sample rate; otherwise resample it to 16k before feeding it to the model
speech, rate = librosa.load("xxx.wav", sr=16000)
speech = librosa.util.fix_length(speech, size=(16000 * 30))

res = s2t(speech)[0]
print(res)
```

### Example script for long-form ASR/ST

```python
import soundfile as sf
import torch
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

context_len_in_secs = 4   # left and right context when doing buffered inference
batch_size = 32   # depends on the GPU memory
s2t = Speech2TextGreedySearch.from_pretrained(
    "espnet/owsm_ctc_v3.1_1B",
    device='cuda' if torch.cuda.is_available() else 'cpu',
    generate_interctc_outputs=False,
    lang_sym='<eng>',
    task_sym='<asr>',
)

speech, rate = sf.read(
    "xxx.wav"
)

text = s2t.decode_long_batched_buffered(
    speech,
    batch_size=batch_size,
    context_len_in_secs=context_len_in_secs,
)
print(text)
```

### Example of CTC forced alignment using `ctc-segmentation`

CTC segmentation can be efficiently applied to audio of an arbitrary length.

```python
import soundfile as sf
from espnet2.bin.s2t_ctc_align import CTCSegmentation
from espnet_model_zoo.downloader import ModelDownloader

# Download model first
d = ModelDownloader()
downloaded = d.download_and_unpack("espnet/owsm_ctc_v3.2_ft_1B")   # "espnet/owsm_ctc_v3.1_1B"

aligner = CTCSegmentation(
    **downloaded,
    fs=16000,
    ngpu=1,
    batch_size=32,    # batched parallel decoding; reduce it if your GPU memory is smaller
    kaldi_style_text=True,
    time_stamps="auto",     # "auto" can be more accurate than "fixed" when converting token index to timestamp
    lang_sym="<eng>",
    task_sym="<asr>",
    context_len_in_secs=2,  # left and right context in buffered decoding
)

speech, rate = sf.read(
    "./test_utils/ctc_align_test.wav"
)
print(f"speech duration: {len(speech) / rate : .2f} seconds")
text = """
utt1 THE SALE OF THE HOTELS
utt2 IS PART OF HOLIDAY'S STRATEGY
utt3 TO SELL OFF ASSETS
utt4 AND CONCENTRATE ON PROPERTY MANAGEMENT
"""

segments = aligner(speech, text)
print(segments)
```
