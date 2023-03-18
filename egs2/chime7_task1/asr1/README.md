# CHiME-7 DASR (CHiME-7 Task 1)

### Distant Automatic Speech Transcription with Multiple Devices in Diverse Scenarios

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/chimechallenge.svg?style=social&label=Follow%20%40chimechallenge)](https://twitter.com/chimechallenge)
[![Slack][slack-badge]][slack-invite]


---

#### If you want to participate please fill this [Google form](https://forms.gle/vbk4gpF77hP5LgKM8) (one contact person per-team only, we will use it to handle submission see [official website, submission page](https://www.chimechallenge.org/current/task1/submission))
### Sections
1. <a href="#description">Short Description </a>
2. <a href="#data_creation">Data Download and Creation</a>
3. <a href="#baseline">Baseline System</a>
4. <a href="#eval_script">Evaluation</a>
5. <a href="#common_issues">Common Issues</a>
6. <a href="#reference">References</a>

## <a id="description">1. Short Description  </a>

<img src="https://www.chimechallenge.org/current/task1/images/task_overview.png" width="450" height="230" />

This CHiME-7 Challenge Task inherits directly from the previous [CHiME-6 Challenge](https://chimechallenge.github.io/chime6/).
Its focus is on distant automatic speech transcription and segmentation with multiple recording devices.

The goal of each participant is to devise an automated system that can tackle this problem, and is able to generalize across different array topologies and different application scenarios: meetings, dinner parties and interviews.

Participants can possibly exploit commonly used open-source datasets (e.g. Librispeech) and pre-trained models. In detail, this includes popular self-supervised representation (SSLR) models (see [Rules Section](https://www.chimechallenge.org/current/task1/rules) for a complete list).

---

**See the [official website](https://www.chimechallenge.org/current/task1/index) to learn more !**
<br>
**All participants will be invited to present their submission to our [Interspeech 2023 Satellite Workshop](https://www.chimechallenge.org/current/workshop/index).**

### <a id="reach_us">Any Question/Problem ? Reach us !</a>

If you are considering participating or just want to learn more then please join the <a href="https://groups.google.com/g/chime5/">CHiME Google Group</a>. <br>
We have also a [CHiME Slack Workspace][slack-invite].<br>
Follow us on [Twitter][Twitter], we will also use that to make announcements. <br>



## <a id="installation">2. Installation </a>

**NOTE, if you don't want to use the baseline at all**, you can
skip this following procedure and go directly to Section 2.
For data generation, only the scripts local/data/gen_task1_data.sh, local/data/gen_task1_data.py and
local/data/generate_chime6_data.sh are used. You can git clone ESPNet, and run these without
ESPNet installation as their dependencies are minimal (`local/data/generate_chime6_data.sh` needs [sox](https://sox.sourceforge.net/) however).

### <a id="espnet_installation">2.1 ESPNet Installation  </a>

Firstly, clone ESPNet. <br/>
```bash
git clone https://github.com/espnet/espnet/
```
Next, ESPNet must be installed, go to espnet/tools. <br />
This will install a new environment called espnet with Python 3.9.2
```bash
cd espnet/tools
./setup_anaconda.sh venv "" 3.9.2
```
Activate this new environment.
```bash
source ./venv/bin/activate
```
Then install ESPNet with Pytorch 1.13.1 be sure to put the correct version for **cudatoolkit**.
```bash
make TH_VERSION=1.13.1 CUDA_VERSION=11.6
```
If you plan to train the ASR model, you would need to compile Kaldi. Otherwise you can
skip this step. Go to the `kaldi` directory and follow instructions in `INSTALL`.
```bash
cd kaldi
cat INSTALL
```
Finally, get in this recipe folder and install other baseline required packages (e.g. lhotse) using this script:
```bash
cd ../egs2/chime7_task1/asr1
./local/install_dependencies.sh
```
You should be good to go !

**NOTE:** if you encounter any problem have a look at <a href="#common_issues">Section 4. Common Issues</a> here below. <br>
Or reach us, see <a href="#reach_us">Section 2.</a>


## <a id="data_creation">2. Data Download and Creation </a>
See Data page in [CHiME-7 DASR webpage](https://www.chimechallenge.org/current/task1/data)
for instructions on how data is structured.
See the [Rules page](https://www.chimechallenge.org/current/task1/rules) too for info about allowed external datasets and pre-trained models.

CHiME-7 DASR makes use of three datasets (but participants can use optionally also allowed external datasets),
these are:

1. CHiME-6 Challenge [1]
2. Amazon Alexa Dinner Party Corpus (DiPCO) [2]
3. LDC Mixer 6 Speech (here we use a new re-annotated version) [3]

Unfortunately, only DiPCo can be downloaded automatically, the others must be
downloaded manually and then processed via our scripts here. <br>
See [Data page](https://www.chimechallenge.org/current/task1/data) for
the instructions on how to obtain the re-annotated Mixer 6 Speech and CHiME-6 data.

### <a id="data_description">2.1 Generating the Data</a>
To generate the data you need to have downloaded and unpacked manually Mixer 6 Speech
and the CHiME-5 dataset as obtained from instructions here [Data page](https://www.chimechallenge.org/current/task1/data).


**Stage 0** of `run.sh` here handles CHiME-7 DASR dataset creation and calls `local/gen_task1_data.sh`. <br>
Note that DiPCo will be downloaded and extracted automatically. <br>
To **ONLY** generate the data you will need to run:
```bash
./run.sh --chime5-root YOUR_PATH_TO_CHiME5 --dipco-root PATH_WHERE_DOWNLOAD_DIPCO \
--mixer6-root YOUR_PATH_TO_MIXER6 --chime6-path PATH_WHERE_STORE_CHiME6 --stage 0 --stop-stage 0
```
If you have already CHiME-6 data you can use that without re-creating it from CHiME-5.
```bash
./run.sh --chime6-root YOUR_PATH_TO_CHiME6 --dipco-root PATH_WHERE_DOWNLOAD_DIPCO \
--mixer6-root YOUR_PATH_TO_MIXER6 --stage 0 --stop-stage 0
```
**If you want to run the recipe from data prep to ASR training and decoding**, instead remove the stop-stage flag.
But please you want to take a look at arguments such as `gss_max_batch_dur`
and `asr_batch_size` because you may want to adjust these based on your hardware.
```bash
./run.sh --chime6-root YOUR_PATH_TO_CHiME6 --dipco-root PATH_WHERE_DOWNLOAD_DIPCO \
--mixer6-root YOUR_PATH_TO_MIXER6 --stage 0 --ngpu YOUR_NUMBER_OF_GPUs
```
**We also provide a pre-trained model**, you can run only inference on development set
using:
```bash
./run.sh --chime6-root YOUR_PATH_TO_CHiME6 --dipco-root PATH_WHERE_DOWNLOAD_DIPCO \
--mixer6-root YOUR_PATH_TO_MIXER6 --stage 0 --ngpu YOUR_NUMBER_OF_GPUs \
--use-pretrained popcornell/chime7_task1_asr1_baseline \
--decode_only 1 --gss-max-batch-dur 30-360-DEPENDING_ON_GPU_MEM
```
Note that `gss-max-batch-dur` affects a lot your inference time.
Also note that getting this warning `Discarded recording P56_dipco_S34_431-120171_120933-mdm from AudioSource(type='file', channels=[8]` 
for a single utterance (but multiple recordings) in the DiPCo dev set is fine. <br>
It is due to some recordings in that DiPCo session being shorter.

You should be able to replicate our results detailed in Section 3.1.2 with the
top 80% envelope variance automatic channel selection. <br>
Note that results may differ a little because GSS inference is not deterministic.
We report the logs for running the scoring script with the pre-trained
ASR model in `baseline_logs/inference.log`.
### <a id="data_description">2.2 Quick Data Overview</a>
The generated dataset folder after running the script should look like this:

```
.
├── chime6
│   ├── audio
│   │   ├── dev
│   │   ├── eval
│   │   └── train
│   ├── transcriptions
│   │   ├── dev
│   │   ├── eval
│   │   └── train
│   ├── transcriptions_scoring
│   │   ├── dev
│   │   ├── eval
│   │   └── train
│   └── uem
│       ├── dev
│       ├── eval
│       └── train
├── dipco
│   ├── audio
│   │   ├── dev
│   │   └── eval
│   ├── transcriptions
│   │   ├── dev
│   │   └── eval
│   ├── transcriptions_scoring
│   │   ├── dev
│   │   └── eval
│   └── uem
│       ├── dev
│       └── eval
└── mixer6
    ├── audio
    │   ├── dev
    │   ├── eval
    │   ├── train_call
    │   └── train_intv
    ├── transcriptions
    │   ├── dev
    │   ├── eval
    │   ├── train_call
    │   └── train_intv
    ├── transcriptions_scoring
    │   ├── dev
    │   ├── eval
    │   ├── train_call
    │   └── train_intv
    └── uem
        ├── dev
        ├── eval
        ├── train_call
        └── train_intv
```


**NOTE** <br>
Eval data (which is actually blind only for Mixer6) can be generated by using as an argument
`--gen-eval 1` (note you will need the Mixer6 eval set which will be released [later](https://www.chimechallenge.org/current/dates)).
<br>

---
To find out if the data has been generated correctly, you can run this
script from this directory <br>
If it runs successfully your MD5 checksums are correct. It has not been parallelized so it will take ~30 mins.<br>
MD5 checksums are stored here in this repo in `local/chime7_dasr_md5.json`.
```bash
python ./local/check_data_gen.py -c PATH_TO_CHIME7TASK_DATA
```
When evaluation data is released, please re-check again to be sure:
```bash
python ./local/check_data_gen.py -c PATH_TO_CHIME7TASK_DATA -e 1
```

Additional data description is available in [CHiME-7 DASR website Data page](https://www.chimechallenge.org/current/task1/data).

## <a id="baseline">3. Baseline System</a>

<img src="https://www.chimechallenge.org/current/task1/images/baseline.png" width="450" height="120" />

The baseline system in this recipe is similar to `egs2/chime6` one, which
itself is inherited directly from CHiME-6 Challenge Kaldi recipe for Track 1 [s5_track1](https://github.com/kaldi-asr/kaldi/tree/master/egs/chime6/s5_track1). <br>


It is composed of two modules (with an optional channel selection module):
1. Guided Source Separation (GSS) [5], here we employ the GPU-based version (much faster) from [Desh Raj](https://github.com/desh2608/gss).
2. End-to-end ASR model based on [4], which is a transformer encoder/decoder model trained <br>
with joint CTC/attention [6]. It uses WavLM [7] as a feature extractor.
3. Optional Automatic Channel selection based on Envelope Variance Measure (EV) [8].

### 3.1 Results

#### 3.1.1 Main Track [Work in Progress]
The main track baseline unfortunately is currently WIP, we hope we can finish it
in the next weeks. <br>
It will be based on a TS-VAD diarization model which leverages pre-trained self-supervised
representation. <br>
We apologize for the inconvenience. <br>
The result of the diarization will be fed to this recipe GSS+ASR pipeline.

#### 3.1.2 Acoustic Robustness Sub-Track: Oracle Diarization + ASR
Pretrained model: [popcornell/chime7_task1_asr1_baseline](popcornell/chime7_task1_asr1_baseline) <br>
Detailed decoding results (insertions, deletions etc) are available in `baseline_logs/RESULTS.md` here. <br>

Note that WER figures in `baseline_logs/RESULTS.md` and in the model card in [popcornell/chime7_task1_asr1_baseline](popcornell/chime7_task1_asr1_baseline) will
be different slighly (a bit higher) from the diarization-attributed WER (DA-WER) score we use for ranking. This latter, as cpWER [1], is based on concatenated utterances for each speaker.
The final score is described in Section 4 and is similar to the one used in previous CHiME-6 Challenge except
that it is not permutation invariant now, but the speaker mapping is instead assigned via diarization (hence diarization-assigned WER).

Here we report the results obtained using channel selection (retaining 80% of all channels) prior to performing GSS and decoding with the baseline pre-trained
ASR model. This is the configuration that gave the best results overall on the dev set.

<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>split</th>
    <th>front-end</th>
    <th>DA-WER (%)</th>
    <th>macro DA-WER (%)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>CHiME-6</td>
    <td rowspan="3">dev</td>
    <td>GSS (EV top 80%)</td>
    <td> 32.64 </td>
    <td rowspan="3"> 28.81 </td>
  </tr>
  <tr>
    <td>DiPCo</td>
    <td>GSS (EV top 80%)</td>
    <td> 33.54 </td>
  </tr>
  <tr>
    <td>Mixer-6</td>
    <td>GSS (EV top 80%)</td>
    <td> 20.25 </td>
  </tr>
</tbody>
</table>


Such baseline system would rank third on dev set based on the rules of the past CHiME-6 Challenge
on Track 1 (unconstrained LM).
Results on the evaluation set will be released after the end of the CHiME-7 DASR Challenge. <br>
Note that as explained in  [CHiME-7 DASR website Data page](https://www.chimechallenge.org/current/task1/data)
the evaluation set for CHiME-6 is different from the one in previous edition and is supposed to be 
more challenging. 


## <a id="eval_script"> 4. Evaluation Script </a>
The evaluation protocol is depicted here below:

<img src="https://www.chimechallenge.org/current/task1/images/main_eval.png" width="450" height="230" />

Evaluation is performed as described in the [Task Main Page](https://www.chimechallenge.org/current/task1/index) and needs joint diarization and transcription.
Systems are evaluated by diarization-attributed word error rate (DA-WER). It is a form of speaker-attributed WER (SA-WER),
where the hypothesis for the WER are re-ordered based on the best reordering defined by diarization error rate (DER), concatenated together
and finally WER is accumulated across all the speakers.
It is similar to the CHiME-6 Challenge cpWER but here we use diarization to define the speaker mapping.
<br>
For the acoustic robustness sub-track we skip the re-ordering process as the participants can use oracle diarization
to provide correctly named speaker labels as well as the correct number of speakers.
The final ranking is given by the DA-WER macro-averaged across all three scenarios (sample from evaluation script output):
```bash
###################################################
### Metrics for all Scenarios ###
###################################################
+----+------------+---------------+---------------+----------------------+----------------------+--------+-----------------+-------------+--------------+----------+
|    | scenario   |   num spk hyp |   num spk ref |   tot utterances hyp |   tot utterances ref |   hits |   substitutions |   deletions |   insertions |      wer |
|----+------------+---------------+---------------+----------------------+----------------------+--------+-----------------+-------------+--------------+----------|
|  0 | chime6     |             8 |             8 |                 6644 |                 6644 |  42748 |           11836 |        4297 |         3090 | 0.326472 |
|  0 | dipco      |            20 |            20 |                 3673 |                 3673 |  22125 |            5859 |        1982 |         2212 | 0.33548  |
|  0 | mixer6     |           118 |           118 |                14804 |                14804 | 126617 |           16012 |        6352 |         7818 | 0.20259  |
+----+------------+---------------+---------------+----------------------+----------------------+--------+-----------------+-------------+--------------+----------+
####################################################################
### Macro-Averaged Metrics across all Scenarios (Ranking Metric) ###
####################################################################
+----+---------------+---------------+---------------+----------------------+----------------------+--------+-----------------+-------------+--------------+----------+
|    | scenario      |   num spk hyp |   num spk ref |   tot utterances hyp |   tot utterances ref |   hits |   substitutions |   deletions |   insertions |      wer |
|----+---------------+---------------+---------------+----------------------+----------------------+--------+-----------------+-------------+--------------+----------|
|  0 | macro-average |       48.6667 |       48.6667 |              8373.67 |              8373.67 |  63830 |         11235.7 |     4210.33 |      4373.33 | 0.288181 |
+----+---------------+---------------+---------------+----------------------+----------------------+--------+-----------------+-------------+--------------+----------+
```

It is performed here in stage 4 in `run.sh` and the scoring takes place
in `local/da_wer_scoring.py`.
The output of the scoring script on the acoustic robustness sub-track for the baseline
is reported in `baseline_logs/inference.log`. <br>
Note that it also produces a lot of information useful for debugging in `${asr_exp}/${inference_tag}/scoring`,
e.g. error statistics in form of `csv` files for all sessions and all speakers as well as
reordered JSON hypotheses with diarization-derived speaker mapping. <br>
When diarization error rate is computed we also log
pyannote annotation and error analysis segmentation, for each session.

The main motivation behind the use of this metric is that we want participants
to produce also reasonable timestamps for each utterance.

cpWER [1] or MIMO-WER [9] do not account directly for timestamps. **asclite** can (time cost option) but it requires
per-words boundaries for which the ground truth is difficult to define and obtain on
the scenarios here. Of course, the choice of DA-WER comes also with drawbacks,
e.g. for short meetings with lots of participants and a lot of overlapped speech
diarization may assign a bad permutation. This however would not be a crucial problem in
the scenarios considered (even > 70% DER will often assign the correct permutation based on our early experiments).



## <a id="common_issues"> 5. Common Issues </a>

1. `AssertionError: Torch not compiled with CUDA enabled` <br> for some reason you installed Pytorch without CUDA support. <br>
 Please install Pytorch with CUDA support as explained in [pytorch website](https://pytorch.org/).
2. `ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory: 'YOUR_PATH/espnet/tools/venv/lib/pyth
on3.9/site-packages/numpy-1.23.5.dist-info/METADATA'`. This is due to numpy installation getting corrupted for some reason.
You can remove the site-packages/numpy- folder manually and try to reinstall numpy 1.23.5 with pip.
3. `FileNotFoundError: [Errno 2] No such file or directory: 'PATH2YOURESPNET/espnet/tools/venv/bin/sox'
` during CHiME-6 generation from CHiME-5, `correct_signals_for_clock_drift.py` script: try to install conda sox again, via `conda install -c conda-forge sox`.
4. `ModuleNotFoundError: No module named 's3prl'` for some reason s3prl did not install, run `YOUR_ESPNET_ROOT/tools/installers/install_s3prl.sh`
5. `Command 'gss' not found` for some reason gss did not install, you can run `YOUR_ESPNET_ROOT/tools/installers/install_gss.sh`
7. `wav-reverberate command not found` you need to install Kaldi. go to `YOUR_ESPNET_ROOT/tools/kaldi` and follow the instructions
in `INSTALL`.
8. `WARNING  [enhancer.py:245] Out of memory error while processing the batch` you got out-of-memory (OOM) when running GSS.
You could try changing parameters as `gss_max_batch_dur` and in local/run_gss.sh `context-duration`
(this latter could degrade results however). See local/run_gss.sh for more info.
9. Much worse WER than baseline and you are using `run.pl`. Check the GSS results, GSS currently does not work well
if you use multi-gpu inference and your GPUs are in shared mode. You need to run `set nvidia-smi -c 3`.


## Memory Consumption (Useful for SLURM etc.)

Figures kindly reported by Christoph Boeddeker, running this baseline code
on Paderborn Center for Parallel Computing cluster (which uses SLURM).
These figures could be useful to anyone that uses job schedulers and clusters
for which resources are assigned strictly (e.g. job killed if it exceed requested
memory resources).

Used as default:
 - train: 3G mem
 - cuda: 4G mem (1 GPU)
 - decode: 4G mem

GSS:
 - nj=8 (default 4)
 - time 45 h per job (but might be faster)
 - num_threads=5 (1 for main process, 4 for DataLoader processes)
 - Mem: 2 GB per core or 10 GB per job. In SLURM it is --mem 2GB, since mem means mem-per-core in SLURM.

scripts/audio/format_wav_scp.sh:
 - Some spikes to the range of 15 to 17 GB

`${python} -m espnet2.bin.${asr_task}_inference${inference_bin_tag`}:
 - Few spikes to the 9 to 11 GB range.



## Acknowledgements

We would like to thank Naoyuki Kamo for his precious help, Christoph Boeddeker for
reporting many bugs and the memory consumption figures and feedback for evaluation script.


## <a id="reference"> 6. References </a>

[1] Watanabe, S., Mandel, M., Barker, J., Vincent, E., Arora, A., Chang, X., et al. CHiME-6 challenge: Tackling multispeaker speech recognition for unsegmented recordings. <https://arxiv.org/abs/2004.09249> <br>
[2] Van Segbroeck, M., Zaid, A., Kutsenko, K., Huerta, C., Nguyen, T., Luo, X., et al. (2019). DiPCo--Dinner Party Corpus. <https://arxiv.org/abs/1909.13447> <br>
[3] Brandschain, L., Graff, D., Cieri, C., Walker, K., Caruso, C., & Neely, A. (2010, May). Mixer 6. In Proceedings of the Seventh International Conference on Language Resources and Evaluation (LREC'10). <br>
[4] Chang, X., Maekaku, T., Fujita, Y., & Watanabe, S. (2022). End-to-end integration of speech recognition, speech enhancement, and self-supervised learning representation. <https://arxiv.org/abs/2204.00540> <br>
[5] Boeddeker, C., Heitkaemper, J., Schmalenstroeer, J., Drude, L., Heymann, J., & Haeb-Umbach, R. (2018, September). Front-end processing for the CHiME-5 dinner party scenario. In CHiME5 Workshop, Hyderabad, India (Vol. 1). <br>
[6] Kim, S., Hori, T., & Watanabe, S. (2017, March). Joint CTC-attention based end-to-end speech recognition using multi-task learning. Proc. of ICASSP (pp. 4835-4839). IEEE. <br>
[7] Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., Chen, Z., ... & Wei, F. (2022). Wavlm: Large-scale self-supervised pre-training for full stack speech processing. IEEE Journal of Selected Topics in Signal Processing, 16(6), 1505-1518. <br>
[8] Wolf, M., & Nadeu, C. (2014). Channel selection measures for multi-microphone speech recognition. Speech Communication, 57, 170-180.
[9] von Neumann T, Boeddeker C, Kinoshita K, Delcroix M, Haeb-Umbach R. On Word Error Rate Definitions and their Efficient Computation for Multi-Speaker Speech Recognition Systems. arXiv preprint arXiv:2211.16112. 2022 Nov 29.

[slack-badge]: https://img.shields.io/badge/slack-chat-green.svg?logo=slack
[slack-invite]: https://join.slack.com/t/chime-fey5388/shared_invite/zt-1oha0gedv-JEUr1mSztR7~iK9AxM4HOA
[twitter]: https://twitter.com/chimechallenge<h2>References</h2>
