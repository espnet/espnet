# ESPnet-AVSR

## Introduction
This repository contains an implementation of end-to-end (E2E) audio-visual speech recognition (AVSR) based on the ESPnet ASR toolkit. The new fusion strategy follows the paper "Fusing information streams in end-to-end audio-visual speech recognition." (https://ieeexplore.ieee.org/document/9414553) [[1]](#literature). A broad range of reliability measures are used to help the integration model improve the performance of the AVSR model. We use two large-vocabulary datasets, the Lip Reading Sentences 2 and 3 corpora for all our experiments.
In addition, this project also contains an audio-only model for comparison.

## Table of Contents 
- [Installation](#installation-of-required-packages)
  * [Requirements](#requirements)
- [Project Structure](#project-structure)
  * [Basics](#project-structure)
  * [AVSR1](#detailed-description-of-avsr1)
- [Usage of the scripts](#running-the-script)
  + [Notes](#notes)


## Installation of required packages

### Requirements

For installation, approximately 40GB of free disk space is needed. avsr1/run.sh stage 0 installs all required packages in avsr1/local/installations:
    
**Required Packages:**
1. ESPNet: https://github.com/espnet/espnet
1. OpenFace: https://github.com/TadasBaltrusaitis/OpenFace
2. DeepXi: https://github.com/anicolson/DeepXi
3. Vidaug: https://github.com/okankop/vidaug

<!-- **Prerequirements:**
The following packages needs to be installed in advance to be able to run the scripts:
1. Git: 
```console 
foo@bar:~$ sudo apt-get install git
```
2. Python-venv package:
```console 
foo@bar:~$ sudo apt-get install python3-venv
```
3. Python-dev packages (development package):
```console 
foo@bar:~$ sudo apt-get install python3-dev # for python 3.+
foo@bar:~$ sudo apt-get install python-dev # for python 2.+ (optional)
```
### Files
The following shell scripts in the installation directory (<code>install/</code>) are important 
 * [install_espnet.sh](install/install_espnet.sh): Script to install ESPnet. For more information see [Option 1](#option-1-installation-scripts-for-every-package-preferred).
 * [install_openface.sh](install/install_openface.sh): Script to install OpenFace. For more information see [Option 1](#option-1-installation-scripts-for-every-package-preferred).
 * [install_deepxi.sh](install/install_deepxi.sh): Script to install DeepXi. For more information see [Option 1](#option-1-installation-scripts-for-every-package-preferred).
 * [install_videoaug.sh](install/install_videoaug.sh): Script to install Videoaug. For more information see [Option 1](#option-1-installation-scripts-for-every-package-preferred).
 * [install_avsr.sh](install/install_avsr.sh): Script to install all packages. For more information see [Option 2](#option-2-installation-script-for-all-packages-not-preferred). 

### Installation
The script install_avsr.sh installs all the requirements (OpenFace, DeepXi, ffmpeg). Be careful in using the script, it was only tested on Ubuntu 20.04 but should also work on Ubuntu 18.04. We assume no liability for damage to your equipment. If any of the requirements are already installed, please install the other packages manually by following the provided links or follow the procedure for this specific package in the installation script. It was tried to build the script so that the Software OpenFace, DeepXI, and ffmpeg can be installed separately. 
Furthermore, if there is an error, the installation stops and you need to follow the error messages to fix them. Some common error sources with mitigation are listed in the section [Common Errors](#common-errors).

>However, the **preferred option is to run each of the scripts** (`install_openface.sh`, `install_deepxi.sh`) **separately** ([Option 1](#option-1-installation-scripts-for-every-package-preferred)). In this case it is easier to react in case of errors.

#### Option 1: Installation scripts for every package (preferred)
1. Install ESPnet. During installation, the user must specify whether CUDA is installed and whether CUDA should be used. The script was written for CUDA 10.0, assuming the default installation path (`/usr/local/cuda-10.0`) for CUDA 10.0. **If the CUDA installation path is different or a different CUDA version should be used, the variable CUDA_PATH must be adjusted in the file install_espnet.sh.** Run the script using simple bash command. 
```console
foo@bar:~/install$ bash install_espnet.sh
```
2. Install OpenFace
```console
foo@bar:~/install$ bash install_openface.sh
```
If there are problems regarding cmake, please refer to [CMake errors](#cmake-version-not-supported) for further instructions.

3. Install DeepXi
```console
foo@bar:~/install$ bash install_deepxi.sh
```
4. Install Videoaug (ESPnet must be installed in advance because Videoaug is installed in ESPnet environment)
```console
foo@bar:~/install$ bash install_deepxi.sh
```

#### Option 2: Installation script for all packages (not preferred)
To install all packages at once, run the command
```console
foo@bar:~/install$ bash install_avsr.sh
```
You can select which packages should be installed while running the bash script.
If there are problems regarding CMake, please refer to [CMake errors](#cmake-version-not-supported) for further instructions.

### Common Errors
#### CMake version not supported
##### Using provided script for Installing CMake 3.10.2
Some errors are associated with a non matching CMake version. The minimum required version is CMake 3.10.2. The script avsr_install.sh can install CMake 3.10.2 with the following command automatically and uses this version for the current session:
```console
foo@bar:~/install$ bash install_avsr.sh INSTALL_CMAKE # or for openface script: bash install_openface.sh INSTALL_CMAKE 
```
To install the CMake version 3.10.2 and use it as your standard CMake version, please use the script with the option:
```console
foo@bar:~/install$ bash install_avsr.sh INSTALL_CMAKE_PERMANENT # or for openface script: bash install_openface.sh INSTALL_CMAKE_PERMANENT
```
##### Using arbitrary CMake versions
If you want to install CMake manually for an arbirtray version, please perform the following steps. Be carefull to replace the version used here (3.10.2) with the correct CMake version you want to install.
###### 1. Open a terminal
###### 2. Download and unpack files: 
Enter the following command to download the source code. Replace with wanted version. 
Here Version 3.10.2 is downloaded.
```console
foo@bar:~$ wget https://github.com/Kitware/CMake/releases/download/v3.10.2/cmake-3.10.2.tar.gz
foo@bar:~$ tar -zxvf cmake-3.10.2.tar.gz 
foo@bar:~$ rm cmake-3.10.2.tar.gz
```
###### 3. Installation
Execute the following steps to perform the installation:
```console
foo@bar:~$ cd cmake-3.10.2
foo@bar:~/cmake-3.10.2$ ./bootstrap
foo@bar:~/cmake-3.10.2$ make
foo@bar:~/cmake-3.10.2$ sudo make install
foo@bar:~/cmake-3.10.2$ cd ..
```
###### 4. Add to path (temporal/permanently)
If you want to add the path temporal for this terminal session, execute:
```console
foo@bar:~$ export PATH="`pwd`/cmake-3.10.2/bin:$PATH"
```

Otherwise, if this CMake version should be the standard CMake version, the .bashrc file needs to be edited. In your home directory open a terminal:
```console
foo@home:~$
```
Open the .bashrc file with a text editor, e.g. nano:
```console
foo@home:~$ sudo nano .bashrc
```
Add the following line to the .bashrc file (replace path-to-cmake with the CMake installation path, e.g. /home/foo/cmake-3.10.2):
```console
foo@home:~$ export PATH="/path-to-cmake/bin:$PATH"
```
Save and close the file. For updating purposes, reload the .bashrc settings, execute:
```console
foo@home:~$ source ~/.bashrc
```

**The current CMake version used can always be checked with the command:**
The current version used can also be checked with
```console
foo@home:~$ cmake --version
```
#### GCC or G++ Error
Some packages requires a specific GCC or G++ version. To install and use multiple GCC or G++ versions, open a terminal and execute:
```console
foo@home:~$ sudo apt-get install gcc-7 g++-7 gcc-8 g++-8 gcc-9 g++-9
foo@home:~$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 7
foo@home:~$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 7
foo@home:~$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
foo@home:~$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
foo@home:~$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
foo@home:~$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
```
This installs different gcc versions and creates a list of multiple GCC and G++ compiler version.
To check and select the available versions, for the GCC compiler run
```console
foo@home:~$ sudo update-alternatives --config gcc
```
and for the G++ compiler run
```console
foo@home:~$ sudo update-alternatives --config g++
```
The current version used can also be checked with
```console
foo@home:~$ gcc --version
foo@home:~$ g++ --version
```
Thanks to: https://linuxconfig.org/how-to-switch-between-multiple-gcc-and-g-compiler-versions-on-ubuntu-20-04-lts-focal-fossa

#### DeepXi getting stuck at grpcio
While installing DeepXi, it could be possible that the tensorflow installation is getting stuck in the process 
```console
Running setup.py bdist_wheel for grpcio
```
This might be due to the fact that the pip version needs to be upgraded. To upgrade pip, please activate the DeepXi environment:
```console
foo@bar:~$ source ~/venv/DeepXi/bin/activate
(DeepXi) foo@bar: pip3 install --upgrade pip
(DeepXi) foo@bar: deactivate
```
Now, rerun the DeepXi installation procedure (e.g. via install_deepxi.sh script)-->

## Project structure
The main folder <code>avsr1/</code>, contains the code for the audio-visual speech recognition system, also trained on the LRS2 [[2]](#literature) dataset together with the LRS3 dataset (https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html) [[3]](#literature). It follows the basic ESPnet structure. 
The main code for the recognition system is the <code>run.sh</code> script. In the script, the workflow of the systems is performed in multiple stages:

|                                  AVSR                       |
|-------------------------------------------------------------|
| Stage 0: Install required packages                     |
| Stage 1: Data Download and preparation                     |
| Stage 2: Audio augmentation                                 | 
| Stage 3: MP3 files and Feature Generation                   |
| Stage 4: Dictionary and JSON data preparation               | 
| Stage 5: Reliability measures generation                    |
| Stage 6: Language model trainin                             |
| Stage 7: Training of the E2E-AVSR model and Decoding        |


<!--The folder structure for both systems is basically:
* <code>conf/</code>: contains configuration files for the training, decoding, and feature extraction 
* <code>data/</code>: directory for storing data
* <code>exp/</code>: log files, model parameters, training results
* <code>fbank/</code>: speech feature binary files, e.g., ark, scp
* <code>dump*/</code> : ESPnet meta data for tranining, e.g., json, hdf5 
* <code>local/</code>: Contains local runtime scripts for data processing, data augmentation and own written functions (e.g. face recognition in the AVSR system) that are not part of the ESPnet standard processing scripts. During the training stage, a symbolic link is built to the ESPnet. After training, the link will be deleted.  
* <code>steps/</code>: helper scripts from ESPnet (Kaldi)
* <code>utils/</code>: helper scripts from ESPnet (Kaldi) -->
  
<!-- ### Detailed description of ASR1:
##### Stage -1: Data Download
  * Strictly considered not a separate stage, since the data set must be downloaded in advance by yourself. For downloading the dataset, please visit 'https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html/' [[2]](#literature)
  * You will need to sign a data sharing agreement with BBC Research & Development before getting access
  * After downloading, please edit <code>path.sh</code> file and assign the dataset directory path to the <code>DATA_DIR</code> variable
 
##### Stage 0: Data Preparation in Kaldi-Style
  * For every dataset part (pretrain, train, test, validate), prepare the data in Kaldi-Style
  * More information about Kaldi-Style: https://kaldi-asr.org/doc/data_prep.html
  * Segmentation: If the variable <code>segment</code> is true, the data in the pretrain set will be segmented into files with length of 5s to restrict the length of the data
  * Generates the text, utt2spk and wav.scp files 

##### Stage 1: Feature Generation
  * Generate the fillter bank features, by default 80-dimensional filter banks with pitch on each frame
  * Cepstral mean and variance normalization

##### Stage 2: Dictionary and JSON data preparation
  * prepare a dictionary and save the data prepared in the previous steps as .json files
  * If a pretrained language model is used, the dictionary data is replaced

##### Stage 3: Language Model Trainingg
  * Train your own language model on the librispeech dataset (https://www.openslr.org/11/) or use a pretrained language model
  * It is possible to skip the language model and use the system without an external language model. For this, just remove the rnnlm from the decoding stage (5)

##### Stage 4: Training
  * Training of the ASR E2E system by using pretrain and train set

##### Stage 5: Decoding
  * Decoding of the test and validation set-->
  
### Detailed description of AVSR1:

##### Stage 0: Packages installations
  * Install the required packages: ESPNet, OpenFace, DeepXi, Vidaug in avsr1/local/installations. To install OpenFace, you will need sudo right.

##### Stage 1: Data preparation
  * The data set LRS2 [2] must be downloaded in advance by yourself. For downloading the dataset, please visit https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html/ [2]. You will need to sign a data-sharing agreement with BBC Research & Development before getting access. After downloading, please edit <code>path.sh</code> file and assign the dataset directory path to the <code>DATA_DIR</code> variable
  * The same applies to the LRS3 dataset https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html [3]. After downloading, please edit <code>path.sh</code> file and assign the dataset directory path to the <code>DATALRS3_DIR</code> variable
  * Download the Musan dataset for audio data augmentation and save it under <code>${MUSAN_DIR}</code> directory
  * Download Room Impulse Response and Noise Database (RIRS-Noises) and save it under <code>RIRS_NOISES/</code> directory
  * Run <code>audio_data_prep.sh</code> script: Create file lists for the given part of the Dataset, prepare the Kaldi files
  * Dump useful data for training 
  
##### Stage 2: Audio Augmentation
  * Augment the audio data with RIRS Noise
  * Augment the audio data with Musan Noise
  * The augmented files are saved under data/audio/augment whereas the clear audio files can be found in data/audio/clear for all the used datasets (Test, Validation(Val), Train and optional Pretrain)
  
##### Stage 3: Feature Generation
  * Make augmented MP3 files
  * Generate the fbank and mfcc features for the audio signals. By default, 80-dimensional filterbanks with pitch on each frame are used
  * Compute global Cepstral mean and variance normalization (CMVN). This computes goodness of pronunciation (GOP) and extracts phone-level pronunciation features for mispronunciations detection tasks (https://kaldi-asr.org/doc/compute-cmvn-stats_8cc.html).
  
##### Stage 4: Dictionary and JSON data preparation
  * Build Dictionary and JSON Data Preparation
  * Build a tokenizer using Sentencepiece: https://github.com/google/sentencepiece

##### Stage 5: Reliability measures generation
  * Stage 5.0: Creat dump file for MFCC features
  * Stage 5.1: Video augmentation with Gaussian blur and salt&pepper noise
  * Stage 5.2: OpenFace face recognition for facial recognition (especially the mouth region, for further details see documentation in avsr1/local folder )
  * Stage 5.3: Extract video frames
  * Stage 5.4: Estimate SNRs using DeepXi framework
  * Stage 5.5: Extract video features by pretrained video feature extractor [[4]](#literature)
  * Stage 5.6: Make video .ark files
  * Stage 5.7: Remake audio and video dump files
  * Stage 5.8: Split test decode dump files by different signal-to-noise ratios
  
##### Stage 6: Language Model Training
  * Train your own language model on the librispeech dataset (https://www.openslr.org/11/) or use a pretrained language model
  * It is possible to skip the language model and use the system without an external language model. 
  
##### Stage 7: Network Training
  * Train audio model
  * Pretrain video model
  * Finetune video model
  * Pretrain av model
  * Finetune av model (model used for decoding)
  
##### Other important references:
  * Explanation of the CSV-file for OpenFace: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format#featureextraction


## Running the script 
The runtime script is the script **run.sh**. It can be found in <code>avsr1/</code> directory.
> Before running the script, please download the LRS2 (https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) [[2]](#literature) and LRS3 (https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html) [[3]](#literature) datasets by yourself and save the download paths to the variables <code>DATA_DIR</code> (LRS2 path) and <code>DATALRS3_DIR</code> (LRS3 path) inside <code>run.sh</code> file.
  
### Notes
Due to the long runtime, it could be useful to run the script using screen command in combination with monitoring in a terminal window and also redirect the output to a log file. 

Screen is a terminal multiplexer which means that you can start any number of virtual terminals inside the current terminal session. The advantage is, that you can detach virtual terminals so that they are running in the background. Furthermore, the processes keep still running, even if you are closing the main session or close an ssh connection if you are working remote on a server.
Screen can be installed from the official package repositories via
```console
foo@bar:~$ sudo apt install screen
```
As an example, to redirect the output into a file named "log_run_sh.txt", the script could be started with:
```console
foo@bar:~/avsr1$ screen bash -c 'bash run.sh |& tee -a log_run_sh.txt'
```
This will start a virtual terminal session, which is executing and monitoring the run.sh file. The output is printed to this session as well as saved into the file "log_run_sh.txt". You can leave the monitoring session by simply pressing <code>ctrl+A+D</code>. If you want to return to the process, simply type 
```console
foo@bar:~$ screen -ls
```
into a terminal to see all running screen processes with their corresponding ID. Then execute
```console
foo@bar:~$ screen -r [ID]
```
to return to the process.
Source: https://wiki.ubuntuusers.de/Screen/

***
### Literature

[1] W. Yu, S. Zeiler and D. Kolossa, "Fusing Information Streams in End-to-End Audio-Visual Speech Recognition," ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 3430-3434, doi: 10.1109/ICASSP39728.2021.9414553.

[2] T. Afouras, J. S. Chung, A. Senior, O. Vinyals, A. Zisserman <br>
Deep Audio-Visual Speech Recognition  
arXiv: 1809.02108

[3] T. Afouras, J. S. Chung, A. Zisserman <br>
LRS3-TED: a large-scale dataset for visual speech recognition  
arXiv preprint arXiv: 1809.00496 

[4] S.  Petridis,   T.  Stafylakis,   P.  Ma,   G.  Tzimiropoulos,   andM.  Pantic,    “Audio-visual  speech  recognition  with  a  hybridCTC/Attention architecture,”   in IEEE SLT. IEEE, 2018.

