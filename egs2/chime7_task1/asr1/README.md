# CHiME-Dive (CHiME-7 Task 1)
### Distant Automatic Speech Transcription with Multiple Devices in Diverse Scenarios

---
## 1. Short Description 

<img src="https://www.chimechallenge.org/current/task1/images/task_overview.png" width="450" height="230" />

## Any Question/Problem ? Reach us !


## 2. Installation

**NOTE, if you don't want to use the baseline at all**, you can 
skip this following procedure and go directly to Section 2.
For data generation, only the scripts create_dataset.sh, local/data/generate_data.py and 
local/data/generate_chime6_data.sh are needed, you can git clone ESPNet, and run these without
ESPNet installation as their dependencies are minimal (`local/data/generate_chime6_data.sh` needs [sox][SoX] however). 

## ESPNet Installation  

First step, clone ESPNet and checkout the correct git commit hash. <br />
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
Activate this new environment (you may want to do this in each new terminal panel).
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

**NOTE:** if you encounter any problem have a look at <a href="#common_issues">Section 4. Common Issues</a> here below.


## 2. Data Download and Creation
See Data Section in [CHiME-7 Task 1 webpage][chime7_task1_webpage] for additional info about the dataset
and the **rules about the data and pre-trained models**. 

Unfortunately, only 




```bash
find chime7_task1 -type f -exec md5sum {} \; | sort -k 2 | md5sum

```
In our case returns:
`f1aaa4e2ec7b95ea6834873ab0221577`


### 2.1 Quick Data Description

```

```

## 3. Baseline System 


### 3.1 Results 


#### 3.1.1 Main Track [Work in Progress]

#### 3.1.2 Sub-Track 1: Oracle Diarization + ASR

Dataset | **microphone** | **cpWER**  |  
--------|--------|------------|
CHiME-6 dev | GSS-all | 33         | 
DiPCo dev | GSS-all| 33         | 
Mixer6 Speech dev | GSS-all | 33         | 


## 4. Evaluation Script [Work in Progress]


## 5. Common Issues

1. `AssertionError: Torch not compiled with CUDA enabled` 
2. `sox: command not found` 
3. `ffmpeg: command not found` 


[chime7_task1_webpage]: 
[sox]:
[google_group]: 
[gpu_gss]:
[gss]: 


## 6. References





