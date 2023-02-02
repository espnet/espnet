# CHiME-Dive (CHiME-7 Task 1)

### Distant Automatic Speech Transcription with Multiple Devices in Diverse Scenarios

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

**NOTE: No registration is required at this stage !** 

### <a id="reach_us">Any Question/Problem ? Reach us !</a>

Consider joining the CHiME Google Group and our Slack ! 



## <a id="installation">2. Installation </a>

**NOTE, if you don't want to use the baseline at all**, you can 
skip this following procedure and go directly to Section 2.
For data generation, only the scripts create_dataset.sh, local/data/generate_data.py and 
local/data/generate_chime6_data.sh are needed, you can git clone ESPNet, and run these without
ESPNet installation as their dependencies are minimal (`local/data/generate_chime6_data.sh` needs [sox][SoX] however). 

### <a id="espnet_installation">2.1 ESPNet Installation  </a> 

First step, clone ESPNet and checkout the correct git commit hash. <br/>
(checkout is important to ensure following ESPNet updates will not break the baseline code)
```bash
git clone https://github.com/espnet/espnet/
```
Next ESPNet must be installed, go to espnet/tools. <br />
This will install a new environment called espnet with Python 3.9.2
```bash
cd espnet
bash ./tools/setup_anaconda.sh venv espnet 3.9.2
```
Activate this new environment.
```bash
source ./tools/venv/bin/activate
conda activate espnet
```
Then install ESPNet with Pytorch 1.13.1 be sure to put the correct version for **cudatoolkit**. 
```bash
make TH_VERSION=1.13.1 CUDA_VERSION=11.6
```
Finally, install other baseline required packages (e.g. lhotse) using this script: 
```bash
./local/install_dependencies.sh
```
You should be good to go !

**NOTE:** if you encounter any problem have a look at <a href="#common_issues">Section 4. Common Issues</a> here below. <br>
Or reach us, see <a href="#reach_us">Section 2.</a>


## <a id="data_creation">2. Data Download and Creation </a>
See Data Section in [CHiME-7 Task 1 webpage][] [1] for additional info about the dataset
and the **rules about the data and pre-trained models**. 

Unfortunately, only 


---
To find out if the data has been generated correctly, you can run this 
snippet from this directory (assuming you called the challenge dataset dir chime7_task1). <br>
Your MD5 checksum should check out with ours. 
```bash
find chime7_task1 -type f -exec md5sum {} \; | sort -k 2 | md5sum
```
In our case it returns:
`f938b57b3b67cdf7a0c0a25d56c45df2` <br>


### <a id="data_description">2.1 Quick Data Description</a>

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
**Evaluation data and annotation will be released later on June 12th**. 

```json
{
  "end_time": "2.370",
  "start_time": "1.000",
  "words": "alright so ummm",
  "speaker": "P03",
  "session_id": "S01"
}
```

```json
{
  "end_time": "1.370",
  "start_time": "1.000",
  "words": "Alright so ummm [noise]",
  "speaker": "P03",
  "session_id": "S01"
}
```
The `uem` directory contains .uem files (Universal Evalution Map). 
These indicate the start and stop (in seconds) for each session on which evaluation will be performed. 
The other parts are ignored for evaluation (but participants are free to use them). <br>

E.g. an example is given below, for session S02 evaluation is performed from second 40 to 8906 approximatively.
The other parts are not scored. 
```
S02 1 40.600 8906.744
S09 1 65.580 7162.197
```

## <a id="baseline">3. Baseline System</a>


### 3.1 Results 

#### 3.1.1 Main Track [Work in Progress]

#### 3.1.2 Sub-Track 1: Oracle Diarization + ASR

Dataset | **microphone** | **cpWER**  |  
--------|--------|------------|
CHiME-6 dev | GSS-all | 33         | 
DiPCo dev | GSS-all| 33         | 
Mixer6 Speech dev | GSS-all | 33         | 


## <a id="eval_script">4. Evaluation [Work in Progress]


## <a id="common_issues"> 5. Common Issues </a>

1. `AssertionError: Torch not compiled with CUDA enabled` 
2. `sox: command not found` 
3. `ffmpeg: command not found` 



[sox]:
[google_group]: 
[gpu_gss]:
[gss]: 


## <a id="reference"> 6. References </a>

[1] Watanabe, S., Mandel, M., Barker, J., Vincent, E., Arora, A., Chang, X., et al. CHiME-6 challenge: Tackling multispeaker speech recognition for unsegmented recordings. <https://arxiv.org/abs/2004.09249> <br>
[2] Chang, X., Maekaku, T., Fujita, Y., & Watanabe, S. (2022). End-to-end integration of speech recognition, speech enhancement, and self-supervised learning representation. <https://arxiv.org/abs/2204.00540> <br>
[3] Boeddeker, C., Heitkaemper, J., Schmalenstroeer, J., Drude, L., Heymann, J., & Haeb-Umbach, R. (2018, September). Front-end processing for the CHiME-5 dinner party scenario. In CHiME5 Workshop, Hyderabad, India (Vol. 1).



