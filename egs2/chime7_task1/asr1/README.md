# CHiME-7 DASR (CHiME-7 Task 1)

### Distant Automatic Speech Transcription with Multiple Devices in Diverse Scenarios

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/chimechallenge.svg?style=social&label=Follow%20%40chimechallenge)](https://twitter.com/chimechallenge)
[![Slack][slack-badge]][slack-invite]

---
### Sections
1. <a href="#description">Short Description </a>
2. <a href="#data_creation">Data Download and Creation</a>
3. <a href="#baseline">Baseline System</a>
4. <a href="#eval_script">Evaluation</a>
5. <a href="#common_issues">Common Issues</a>
6. <a href="#reference">References</a>

## <a id="description">1. Short Description  </a>

<img src="https://www.chimechallenge.org/current/task1/images/task_overview.png" width="450" height="230" />

This CHiME-7 Challenge Task inherits diretly from the previous [CHiME-6 Challenge](https://chimechallenge.github.io/chime6/). 
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
Follow us on [Twitter][Twitter], we will also use that to make announcements. 


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


Stage 0 of `run.sh` here handles CHiME-7 DASR dataset creation and calls `local/gen_task1_data.sh`. <br>
Note that DiPCo will be downloaded and extracted automatically. <br>
To **ONLY** generate the data you will need to run:

```bash
./run.sh --chime5-root YOUR_PATH_TO_CHiME5 --dipco-root PATH_WHERE_DOWNLOAD_DIPCO \
--mixer6-root YOUR_PATH_TO_MIXER6 --stage 0 --stop-stage 0
```
If you have already CHiME-6 data you can use that without re-creating it from CHiME-5. 
```bash
./run.sh --chime6-root YOUR_PATH_TO_CHiME6 --dipco-root PATH_WHERE_DOWNLOAD_DIPCO \
--mixer6-root YOUR_PATH_TO_MIXER6 --stage 0 --stop-stage 0
```
if you want to run the recipe from data prep to ASR training and decoding, instead remove the stop-stage flag.
But please you want to take a look at arguments such as `ngpu`, `gss_max_batch_dur` 
and `asr_batch_size` because you may want to adjust these based on your hardware. 
```bash
./run.sh --chime6-root YOUR_PATH_TO_CHiME6 --dipco-root PATH_WHERE_DOWNLOAD_DIPCO \
--mixer6-root YOUR_PATH_TO_MIXER6 --stage 0 
```


### <a id="data_description">2.2 Quick Data Overview</a>
The generated dataset folder after running the script should look like this: 
```
chime7_task1
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
│   │   └── dev
│   ├── transcriptions
│   │   └── dev
│   └── transcriptions_scoring
│       └── dev
└── mixer6
    ├── audio
    │   ├── dev
    │   ├── train_calls
    │   └── train_intv
    ├── transcriptions
    │   ├── dev
    │   ├── train_calls
    │   └── train_intv
    └── transcriptions_scoring
        ├── dev
        ├── train_calls
        └── train_intv
```
**NOTE**: eval directories for chime6 are empty at this stage. <br>

---
To find out if the data has been generated correctly, you can run this 
snippet from this directory (assuming you called the challenge dataset dir chime7_task1). <br>
Your MD5 checksum should check out with ours. 
```bash
find chime7_task1 -type f -exec md5sum {} \; | sort -k 2 | md5sum
```
In our case it returns:
`f938b57b3b67cdf7a0c0a25d56c45df2` <br>

Additional description is available in [Data page](https://www.chimechallenge.org/current/task1/data)

## <a id="baseline">3. Baseline System</a>

The baseline system in this recipe is similar to `egs2/chime6` one, which 
itself is inherited direcly from CHiME-6 Challenge Kaldi recipe for Track 1 [s5_track1](https://github.com/kaldi-asr/kaldi/tree/master/egs/chime6/s5_track1). <br>

It is composed of two modules:
1. Guided Source Separation (GSS) [5], here we employ the GPU-based version (much faster) from [Desh Raj](https://github.com/desh2608/gss).
2. End-to-end ASR model based on [4], which is a transformer encoder/decoder model trained <br>
with joint CTC/attention [6]. It uses WavLM [7] as a feature extractor. 

### 3.1 Results 

#### 3.1.1 Main Track [Work in Progress]

#### 3.1.2 Sub-Track 1: Oracle Diarization + ASR

Dataset | **microphone** | **SA-WER** |  
--------|--------|------------|
CHiME-6 dev | GSS-all | TBA        | 
DiPCo dev | GSS-all| TBA        | 
Mixer6 Speech dev | GSS-all | TBA        | 


## <a id="eval_script">4. Evaluation [Work in Progress]


## <a id="common_issues"> 5. Common Issues </a>

1. `AssertionError: Torch not compiled with CUDA enabled` <br> for some reason you installed Pytorch without CUDA support. <br>
 Please install Pytorch with CUDA support as explained in [pytorch website](https://pytorch.org/).
2. `ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory: 'YOUR_PATH/espnet/tools/venv/lib/pyth
on3.9/site-packages/numpy-1.23.5.dist-info/METADATA'`. This is due to numpy installation getting corrupted for some reason.
You can remove the site-packages/numpy- folder manually and try to reinstall numpy 1.23.5 with pip. 


[sox]:
[google_group]: 
[gpu_gss]:
[gss]: 


## <a id="reference"> 6. References </a>

[1] Watanabe, S., Mandel, M., Barker, J., Vincent, E., Arora, A., Chang, X., et al. CHiME-6 challenge: Tackling multispeaker speech recognition for unsegmented recordings. <https://arxiv.org/abs/2004.09249> <br>
[2] Van Segbroeck, M., Zaid, A., Kutsenko, K., Huerta, C., Nguyen, T., Luo, X., et al. (2019). DiPCo--Dinner Party Corpus. <https://arxiv.org/abs/1909.13447> <br>
[3] Brandschain, L., Graff, D., Cieri, C., Walker, K., Caruso, C., & Neely, A. (2010, May). Mixer 6. In Proceedings of the Seventh International Conference on Language Resources and Evaluation (LREC'10). <br>
[4] Chang, X., Maekaku, T., Fujita, Y., & Watanabe, S. (2022). End-to-end integration of speech recognition, speech enhancement, and self-supervised learning representation. <https://arxiv.org/abs/2204.00540> <br>
[5] Boeddeker, C., Heitkaemper, J., Schmalenstroeer, J., Drude, L., Heymann, J., & Haeb-Umbach, R. (2018, September). Front-end processing for the CHiME-5 dinner party scenario. In CHiME5 Workshop, Hyderabad, India (Vol. 1). <br>
[6] Kim, S., Hori, T., & Watanabe, S. (2017, March). Joint CTC-attention based end-to-end speech recognition using multi-task learning. Proc. of ICASSP (pp. 4835-4839). IEEE. <br>
[7] Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., Chen, Z., ... & Wei, F. (2022). Wavlm: Large-scale self-supervised pre-training for full stack speech processing. IEEE Journal of Selected Topics in Signal Processing, 16(6), 1505-1518.


[slack-badge]: https://img.shields.io/badge/slack-chat-green.svg?logo=slack
[slack-invite]: https://join.slack.com/t/chime-fey5388/shared_invite/zt-1oha0gedv-JEUr1mSztR7~iK9AxM4HOA
[twitter]: https://twitter.com/chimechallenge<h2>References</h2>