# CHiME-8 DASR (CHiME-8 Task 1)

### Distant Automatic Speech Transcription with Multiple Devices in Diverse Scenarios

#### If you want to participate see [official challenge website](https://www.chimechallenge.org/current/task1/index) for registration.


### <a id="reach_us">Any Question/Problem ? Reach us !</a>

If you are considering participating or just want to learn more then please join the <a href="https://groups.google.com/g/chime5/">CHiME Google Group</a>. <br>
We have also a [CHiME Slack Workspace][slack-invite], join the `chime-8-dasr` channel there or contact us directly.<br>

- We also have a [Troubleshooting page](./HELP.md).

### <a id="whatisnew">What is new compared to CHiME-7 DASR Baseline ? </a>

- GSS now is much more memory efficient see https://github.com/desh2608/gss/pull/39 (many thanks to Christoph Boeddeker).
- Pyannote EEND segmentation model has been retrained on CHiME-6 + NOTSOFAR1 data. It is a bit better.
- Some bugs have been fixed.

## DASR Data Download and Generation

Data generation is handled here using [chime-utils](https://github.com/chimechallenge/chime-utils). <br>
If you are **only interested in obtaining the data** you should use [chime-utils](https://github.com/chimechallenge/chime-utils) directly. <br>

Data generation and downloading is done automatically in this recipe in stage 0. You can skip it if you have already the data. <br>
Note that Mixer 6 Speech has to be obtained via LDC. See [official challenge website](https://www.chimechallenge.org/current/task1/data). <br>
CHiME-6, DiPCo and NOTSOFAR1 will be downloaded automatically.
## System Description

<img src="https://www.chimechallenge.org/challenges/chime7/task1/images/baseline.png" width="450" height="120" />

The system here is effectively the same as used for the CHiME-7 DASR Challenge (except for the changes mentioned previously). <br>
It is described in detail in the [CHiME-7 DASR paper](https://arxiv.org/abs/2306.13734) and the [website of the previous challenge](https://www.chimechallenge.org/challenges/chime7/task1/baseline). <br>
The system consists of:
1. diarization component based on [Pyannote diarization pipeline](https://huggingface.co/pyannote/speaker-diarization)
   - this is in `diar_asr1` folder
2. Envelope-variance selection [4] + Guided source separation [2] + WavLM-based ASR model [1].
   - this is in `asr1` folder.

## Results

As explained in [official challenge website](https://www.chimechallenge.org/current/task1/index) this year
systems will be ranked according to macro tcpWER [5] across the 4 scenarios (5 s collar). <br>
The 4 scenarios we feature this year are very diverse see ([see website for statistics](https://www.chimechallenge.org/current/task1/index)), and this diversity
significantly complicates speaker counting.


## Reproducing the Baseline

NOTE: **GSS currently does not work well if you use multi-gpu inference and your GPUs are in shared mode** <br>
Please if you use `run.pl` set your GPUs in EXCLUSIVE_PROCESS with `nvidia-smi -i 3 -c 3` where `-i X` is the GPU index.

### Inference-only

If you want to perform inference with the pre-trained models:
- ASR ([HF repo](https://huggingface.co/popcornell/chime7_task1_asr1_baseline))
- Pyannote Segmentation ([HF repo](https://huggingface.co/popcornell/chime7_task1_asr1_baseline))

#### Full-System


Got to `diar_asr1`:
```bash
cd diar_asr1
```
If you have already generated the data via [chime-utils](https://github.com/chimechallenge/chime-utils) and the data is in `/path/to/chime8_dasr`:
```bash
./run.sh --chime8-root /path/to/chime8_dasr --stage 1 --ngpu YOUR_NUMBER_OF_GPUs \
--use-pretrained popcornell/chime7_task1_asr1_baseline \
--run-on dev
```
If you need to generate the data yet.
CHiME-6, DiPCo and NOTSOFAR1 will be downloaded automatically. Ensure you have ~1TB of space in a path of your choice `/your/path/to/download`. <br>
Mixer 6 Speech has to be obtained via LDC and unpacked in a directory of your choice `/your/path/to/mixer6_root`. <br>
Data will be generated in `/your/path/to/chime8_dasr` again choose the most convenient location for you.


```bash
./run.sh --chime8-root /path/to/chime8_dasr \
--download-dir /your/path/to/download \
--mixer6-root /your/path/to/mixer6_root \
--stage 0 --ngpu YOUR_NUMBER_OF_GPUs \
--use-pretrained popcornell/chime7_task1_asr1_baseline \
--run-on dev
```

You can use `--stage` and `--gss-asr-stage` args to resume the inference in whatever step.

#### Oracle-Diarization (or diarization from your own diarizer)

We provide also a GSS + ASR only script to be used with oracle diarization
or you diarizer output if you wish only to work on diarization. <br>
We assume here you have already generated the data and start from stage 1.

If you want to use oracle diarization, go to `asr1`:

```bash
cd asr1

./run.sh --chime8-root /path/to/chime8_dasr --stage 1 --ngpu YOUR_NUMBER_OF_GPUs \
--use-pretrained popcornell/chime7_task1_asr1_baseline \
--run-on dev
```

If you want to use your custom diarization, go to `diar_asr1`:
```bash
cd diar_asr1

./run.sh --chime8-root /path/to/chime8_dasr --stage 3 --ngpu YOUR_NUMBER_OF_GPUs \
--use-pretrained popcornell/chime7_task1_asr1_baseline \
--run-on dev --diarization-dir /path/to/your/diarization/output
```

It is assumed that your diarizer produces JSON manifests (same as CHiME-8 DASR annotation see [data page]())
and these manifests are in `/path/to/your/diarization/output`. <br>
`/path/to/your/diarization/output` should have this structure (you can ignore `.rttms`):

```
├── chime6
│   └── dev
│       ├── S02.json
│       ├── S02.rttm
│       ├── S09.json
│       └── S09.rttm
├── dipco
│   └── dev
│       ├── S28.json
│       ├── S28.rttm
│       ├── S29.json
│       └── S29.rttm
├── mixer6
......
├── notsofar1
```


### Training the ASR model

We assume here you have already generated the data and start from stage 1.
If you want to use retrain the ASR model, go to `asr1` and choose a name for the new model:

```bash
cd asr1

./run.sh --chime8-root /path/to/chime8_dasr --stage 1 --ngpu YOUR_NUMBER_OF_GPUs \
--run-on train --asr-tag YOUR_NEW_ASR_NAME
```

### Fine-Tuning the Pyannote Segmentation Model

We assume here you have already generated the data and start from stage 1.
If you want to fine-tune the segmentation model, go to `diar_asr1` and choose a name for the new model:

```bash
cd diar_asr1

./run.sh --chime8-root /path/to/chime8_dasr --stage 1 --ngpu YOUR_NUMBER_OF_GPUs \
--pyan-ft 1
```

Note that the data preparation for the fine-tuning is done in `diar_asr1/local/pyannote_dprep.py`
and you have also to set up `diar_asr1/local/database.yml` properly to use your own data. <br>
See the [pyannote documentation](https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/training_a_model.ipynb) for more info.

## Acknowledgements

We would like to thank Naoyuki Kamo for his precious help, Christoph Boeddeker for
reporting many bugs and the memory consumption figures and feedback for evaluation script.


## <a id="reference"> 6. References </a>

[1] Chang, X., Maekaku, T., Fujita, Y., & Watanabe, S. (2022). End-to-end integration of speech recognition, speech enhancement, and self-supervised learning representation. <https://arxiv.org/abs/2204.00540> <br>
[2] Boeddeker, C., Heitkaemper, J., Schmalenstroeer, J., Drude, L., Heymann, J., & Haeb-Umbach, R. (2018, September). Front-end processing for the CHiME-5 dinner party scenario. In CHiME5 Workshop, Hyderabad, India (Vol. 1). <br>
[3] Kim, S., Hori, T., & Watanabe, S. (2017, March). Joint CTC-attention based end-to-end speech recognition using multi-task learning. Proc. of ICASSP (pp. 4835-4839). IEEE. <br>
[4] Wolf, M., & Nadeu, C. (2014). Channel selection measures for multi-microphone speech recognition. Speech Communication, 57, 170-180.
[5] von Neumann, T., Boeddeker, C., Delcroix, M., & Haeb-Umbach, R. (2023). MeetEval: A Toolkit for Computation of Word Error Rates for Meeting Transcription Systems. arXiv preprint arXiv:2307.11394. <br>


[slack-badge]: https://img.shields.io/badge/slack-chat-green.svg?logo=slack
[slack-invite]: https://join.slack.com/t/chime-fey5388/shared_invite/zt-1oha0gedv-JEUr1mSztR7~iK9AxM4HOA
[twitter]: https://twitter.com/chimechallenge<h2>References</h2>
