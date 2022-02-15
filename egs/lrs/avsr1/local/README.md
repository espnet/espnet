# File Documentation
The documentation is not finished. There are some files (especially in the subdirectories) without documentation right now.
## Table of Contents 
The documentation for the listed files is given below:
- [finetuneav/](#finetuneav)
- [finetunevideo/](#finetunevideo)
- [lrs3processing/](#lrs3processing)
- [pretrainav/](#pretrainav)
- [pretrainvideo/](#pretrainvideo)
- [trainaudio/](#trainaudio)
- [audio_augmentation.sh](#audio_augmentationsh)
- [audio_augmentation_recog.sh](#audio_augmentation_recogsh)
- [audioaugwav.sh](#audioaugwavsh)
- [audio_data_prep.sh](#audio_data_prepsh)
- [audiodump.py](#audiodump.py)
- [avpretraindecodedump.py](#avpretraindecodedump.py)
- [avpretraindump.py](#avpretraindumppy)
- [avtraindecodedump.py](#avtraindecodedumppy)
- [avtraindump.py](#avtraindumppy)
- [convertsnr.py](#convertsnrpy)
- [creatsegfile.py](#createsegfilepy)
- [extractfeatures.sh](#extractfeaturessh)
- [extractframs.sh](#extractframssh)
- [extractsnr.py](#extractsnrpy)
- [extractsnr.sh](#extractsnrsh)
- [extractvfeatures.py](#extractvfeaturespy)
- [finetune_av.sh](#finetune_avsh)
- [finetune_video.sh](#finetunevideosh)
- [LRS3dataprocessing.sh](#LRS3dataprocessingsh)
- [make_video.py](#make_videopy)
- [Openface.sh](#openfacesh)
- [prepaudio.py](#prepaudiopy)
- [preppretrainaudio.py](#preppretrainaudiopy)
- [pretrain.py](#pretrainpy)
- [pretrain_av.sh](#pretrain_avsh)
- [remakeavjson.py](#remakeavjsonpy)
- [remake_dict.py](#remake_dictpy)
- [segaugaudio.py](#segaugaudiopy)
- [segmentaudio.py](#segmentaudiopy)
- [segvideo.py](#segvideopy)
- [sepaudiovideo.py](#sepaudiovideopy)
- [show_results.sh](#show_resultssh)
- [splitsnr.py](#splitsnrpy)
- [train_audio.sh](#train_audiosh)
- [videoaug.py](#videoaugpy)
- [videodump.py](#videodumppy)
---

### finetuneav/
Folder with data preparation, training and decoding scripts for fine-tuning the audio-visual model

---

### finetunevideo/
Folder with data preparation, training and decoding scripts for fine-tuning the video only model

--- 

### lrs3processing/
Folder with scripts for preparing the LRS3 dataset

---

### pretrainav
Folder with data preparation, training and decoding scripts for pretraining the audio-visual model

---

### pretrainvideo/
Folder with data preparation, training and decoding scripts for pretraining the video only model

---

### trainaudio/
Folder with data preparation, training and decoding scripts for training the audio-only model

---

### audio_augmentation.sh
**Short description:** In this section, we augment the VoxCeleb2 data with reverberation, noise, music, and babble, and combine it with the clean data. <br>
**Parameters:**

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>musan_root=$1</code> | Musan root directory | <code>$MUSAN_DIR</code>, usually '/musan', defined in path.sh |
| <code>dset=$2</code> | Dataset part (Train, Test, Val, pretrain) | <code>$part</code> (Train, Test, Val, pretrain) |


**Explanation:**
- The source directory of the clean audio files is stored in <code>srcdir=data/audio/clean</code>, whereas the augmented data is stored in <code>augdatadir=data/audio/augment</code>
- First step is to download the "Rirs-Noises" dataset (https://www.openslr.org/28/) if not already done
- Afterwards, make a version with reverberated speech and make a reverberated version of the VoxCeleb2 list. Note that we do not add any additive noise here
- Next the MUSAN corpus (http://www.openslr.org/17/) is prepared, which consists of music, speech, and noise. This corpus is suitable for augmentation. Then the data is augmented with musan noise
- Finally Combine reverb, noise, music, and babble into one directory (<code>augdatadir</code>)

---

### audio_augmentation_recog.sh
---

### audioaugwav.sh
**Short description:** This script makes the augmented stream data to real .wav data which one can listen to<br>
**Parameters:**

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>srcdir=$1</code> | Path to augmented files | <code>data/audio/augment</code> |
| <code>savedir=$2</code> | Directory in which to save the augmented .wav files  | <code>$mp3files</code>, 'Dataset_processing/Audioaugments' |
| <code>dset=$3</code> | Dataset part (Train, Test, Val, pretrain) | <code>$part</code> (Train, Test, Val, pretrain) |

**Explanation:**
- Take the augmented input files from <code>$srcdir</code>, build .wav files and save them into <code>$savedir/$part</code>

---

### audio_data_prep.sh
**Short description:** Prepare the audio filelists and make the kaldi files<br>
**Parameters:**

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>sdir=$1</code> | Directory of the dataset | <code>$DATA_DIR</code>, usually '/LRS2', defined in path.sh |
| <code>dset=$2</code> | Dataset part (Train, Test, Val, pretrain) | <code>$part</code> (Train, Test, Val, pretrain) |
| <code>ifsegment=$3</code> | If do segmentation for pretrain set | <code>$ifsegment</code> |
| <code>ifdebug=$4</code> | If debug mode should be used. In debug mode, only $num utterances from pretrain and $num Utts from Train set are used | <code>$ifdebug</code> |
| <code>num=$5</code> | Number of utterances used from pretrain and Train set if $debug is true | <code>$num</code> |
| <code>ifmulticore=$6</code>, default = true | If multi cpu processing should be used | <code>$ifmulticore</code> |


**Explanation:**
- Copy the dataset metadata (filelists) to the <code>$metadir</code> (data/METADATA) and prune the filelists for pretrain set and test set if debug mode is used
- For pretrain set, run <code>preppretrainaudio.py</code>, <code>sepaudiovideo.py</code> and <code>segmentaudio.py</code> 
- For all other sets, run <code>prepaudio.py</code> to build the kaldi files

---

### audiodump.py
**Short description:** Remake audio dump files <br>

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>sys.argv[1]</code> | dumpfile (str), Directory to save dump files |  dump/audio |
| <code>sys.argv[2]</code> | dumpsrcfile(str), Directory to save audio dump files | dump/audio_org |
| <code>sys.argv[3]</code> | dset(str), Which dataset | $dset |
| <code>sys.argv[4]</code> | ifmulticore (boolean), If multi cpu processing should be used | $ifmulticore |

---

### avpretraindecodedump.py
**Short description:** Remake dump files for preatraining <br>

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>sys.argv[1]</code> | dumpfile (str), Directory to save dump files | dump/avpretraindecode |
| <code>sys.argv[2]</code> | dumpaudiofile (str), Directory to save audio dump files | dump/audio_org |
| <code>sys.argv[3]</code> | dumpvideofile (str), Directory to save video dump files | dump/video |
| <code>sys.argv[4]</code> | snrdir (str), Directory with SNR saved as .pt files | $SNRptdir, usually Dataset_processing/SNRs |
| <code>sys.argv[5]</code> | vrmdir (str), Directory with confidence saved as .pt files | $videoframe, usually Dataset_processing/Videodata |
| <code>sys.argv[6]</code> | mfccdumpdir (str), Directory to save mfcc dump files | dump/mfcc |
| <code>sys.argv[6]</code> | dset (str), Which dataset | $dset |
| <code>sys.argv[7]</code> | noisecombination (str), Augumented audio and video noise type | $noisecombination |
| <code>sys.argv[8]</code> | ifmulticore (boolean), If multi cpu processing should be used | $ifmulticore |
				
---

### avpretraindump.py
**Short description:** Remake dump files <br>

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>sys.argv[1]</code> | dumpfile (str), Directory to save dump files |  dump/avpretrain |
| <code>sys.argv[2]</code> | dumpaudiofile (str), Directory to save audio dump files | dump/audio_org |
| <code>sys.argv[3]</code> | dumpvideofile (str), Directory to save video dump files | dump/video |
| <code>sys.argv[4]</code> | snrdir (str), Directory with SNR saved as .pt files | $SNRptdir, usually Dataset_processing/SNRs |
| <code>sys.argv[5]</code> | vrmdir (str), Directory with confidence saved as .pt files | $videoframe, ususally Dataset_processing/Videodata |
| <code>sys.argv[6]</code> | mfccdumpdir (str), Directory to save mfcc dump files | dump/mfcc |
| <code>sys.argv[7]</code> | dset(str), Which dataset | $dset |
| <code>sys.argv[8]</code> | ifmulticore (boolean), If multi cpu processing should be used | $ifmulticore |

---

### avtraindecodedump.py
**Short description:** Remake dump files for decoding <br>

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>sys.argv[1]</code> | dumpfile (str), Directory to save dump files |  dump/avpretrain |
| <code>sys.argv[2]</code> | dumpaudiofile (str), Directory to save audio dump files | dump/audio_org |
| <code>sys.argv[3]</code> | dumpvideofile (str), Directory to save video dump files | dump/video |
| <code>sys.argv[4]</code> | videodir (str), Directory to save video dump files | $videofeature, usually Dataset_processing/Videofeature |
| <code>sys.argv[5]</code> | snrdir (str), Directory with SNR saved as .pt files | $SNRptdir, usually Dataset_processing/SNRs |
| <code>sys.argv[6]</code> | vrmdir (str), Directory with confidence saved as .pt files | $videoframe, ususally Dataset_processing/Videodata |
| <code>sys.argv[7]</code> | mfccdumpdir (str), Directory to save mfcc dump files | dump/mfcc |
| <code>sys.argv[8]</code> | dset(str), Which dataset | $dset |
| <code>sys.argv[9]</code> | noisecombination (str), Augumented audio and video noise type | $noisecombination |
| <code>sys.argv[10]</code> | ifmulticore (boolean), If multi cpu processing should be used | $ifmulticore |

---

### avtraindump.py

**Short description:** Remake dump files <br>

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>sys.argv[1]</code> | dumpfile (str), Directory to save dump files |  dump/avpretrain |
| <code>sys.argv[2]</code> | dumpaudiofile (str), Directory to save audio dump files | dump/audio_org |
| <code>sys.argv[3]</code> | videodir (str), Directory to save video feature files | $videofeature, usually Dataset_processing/Videofeature |
| <code>sys.argv[4]</code> | snrdir (str), Directory with SNR saved as .pt files | $SNRptdir, usually Dataset_processing/SNRs |
| <code>sys.argv[5]</code> | vrmdir (str), Directory with confidence saved as .pt files | $videoframe, ususally Dataset_processing/Videodata |
| <code>sys.argv[6]</code> | mfccdumpdir (str), Directory to save mfcc dump files | dump/mfcc |
| <code>sys.argv[7]</code> | dset(str), Which dataset | $dset |
| <code>sys.argv[8]</code> | ifmulticore (boolean), If multi cpu processing should be used | $ifmulticore |

---

### convertsnr.py
**Short description:** Convert the SNR to PyTorch Tensors (.pt) files<br>
**Parameters:**

| Parameter Name | Function | Calling from extractsnr.sh |
|----------------|----------|---------|
| <code>sys.argv[1]</code>, sourcedir (str) | Source directory of the SNR results extracted by DeepXi (see [extractsnr.sh](#extractsnrsh)) | <code>$savematdir/$dset</code>, 'Dataset_processing/SNRsmat/$dset' |
| <code>sys.argv[2]</code>, savedir (str) | Save directory of the converted SNR as .pt files | <code>$saveptdir/$dset</code>, 'Dataset_processing/SNRs/$dset' |
| <code>sys.argv[3]</code>, ifmulticore (boolean) | If multi cpu processing should be used | <code>$ifmulticore</code> |


---

### creatsegfile.py
**Short description:** Create basis for segmented audio files for LRS3 (Build files with correct filenames, but no segmentation of audio signal done in this script)<br>
**Parameters:**

| Parameter Name | Function |
|----------------|----------|
| <code>sys.argv[1]</code>, sourcedir (str) | srcdir (str): Path to source dir |
| <code>sys.argv[2]</code>, videodir (str) | videodir (str): Path to Corpus |
| <code>sys.argv[3]</code>, dset (str) | Which set. For this code dset is pretrain set |
| <code>sys.argv[4]</code>, ifmulticore (str) | ifmulticore (str), if use multi processes |

---

### extractfeatures.sh
**Short description:** Starts the python script [extractvfeatures.py](#extractvfeaturespy) for extracting video features from video frames<br>
**Parameters:**

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>sdir=$1</code> | Source directory of video frame pictures | <code>$videoframe/$part/Pics</code> |
| <code>savedir=$2</code> | Save directory for features | <code> $videofeature/$part</code> |
| <code>pretrainedmodeldir=$3</code> | Path to pretrained video model | <code>$PRETRAINEDMODEL</code>, usually 'pretrainedvideomodel/Video_only_model.pt', defined in path.sh |
| <code>dset=$4</code> | Dataset part (Train, Test, Val, pretrain) | <code>$part</code> (Test, Train, Val, Pretrain) |
| <code>ifcuda=$5</code> | If CUDA should be used | <code>$ifcuda</code> |
| <code>ifdebug=$6</code>, default = true | If debug mode should be used | <code>$ifdebug</code> |

**Explanation:**
- Main Task is to prepare the handover variables for [extractvfeatures.py](#extractvfeaturespy), that runs a feature extraction for the video features for each video frame
- Main Code: <code>python3 -u local/extractvfeatures.py $sdir $savedir $pretrainedmodeldir $dset $ifcuda $ifdebug</code>

---

### extractframs.sh
**Short description:** Starts the python script [segvideo.py](#segvideopy) for frame extraction<br>
**Parameters:**

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>sdir=$1</code> | Directory of the dataset | <code>$DATA_DIR</code>, usually '/LRS2', defined in path.sh |
| <code> savedir=$2 </code> | Save directory of the Segmented video data for every dataset | <code>$videoframe</code>, 'Dataset_processing/Videodata' |
| <code>csvdir=$3</code> | The directory of the .csv Files, which contains the Face recognition information extracted by OpenFace | <code>$facerecog</code>, 'Dataset_processing/Facerecog' |
| <code>audiodir=$4</code> | The directory in which the clean audio data is saved | <code>data/audio/clean</code> |
| <code>dset=$5</code> | Dataset part (Train, Test, Val, pretrain) | <code>$part</code> (Test, Train, Val, Pretrain) |
| <code>ifsegment=$6</code> | If do segmentation for pretrain set | <code>$ifsegment</code> |
| <code>ifmulticore=$7</code>, default = true | If multi cpu processing should be used | <code>$ifmulticore</code> |


**Explanation:**
- Main Task is to prepare the handover variables for [segvideo.py](#segvideopy), that runs the frame extraction
- Creates the variables <code>sourcedir</code>, <code>Confdir</code>, <code>AUdir</code> and <code>datadir</code>
- Main Code: <code>python3 -u local/segvideo.py $sourcedir $savedir/$dset $csvdir/$dset $audiodir/$dset $dset $ifsegment $ifmulticore</code>

---

### extractsnr.py
**Short decscription:** Refactor the extracted SNR and convert the format of the DeepXi results <br>
**Parameters:**

| Parameter Name | Function |
|----------------|----------|
| <code>sys.argv[1]</code>, srcdir (str) | Source directoy | 
| <code>sys.argv[2]</code>, noisetype (str) | Noise type | 

---

### extractsnr.sh
**Short description:** Extract the SNR for the augmented audio data using DeepXi<br>
**Parameters:**

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>savematdir=$1</code> |  | <code>$SNRdir</code>, 'Dataset_processing/SNRsmat' |
| <code>saveptdir=$2</code> |  | <code>$SNRptdir</code>, 'Dataset_processing/SNRs' |
| <code>srcdir=$3</code> | Path to the audio mp3 files (augmented audio files) | <code>$mp3files</code>, 'Dataset_processing/Audioaugments' |
| <code>dset=$4</code> | Dataset part (Train, Test, Val, pretrain) | <code>$part</code> (Train, Test, Val, pretrain) |
| <code>ifmulticore=true</code>, default=true | If multi cpu processing should be used | <code>$ifmulticore</code> |



**Explanation:**
- Main Task: Extract the SNR using DeepXi Framework (https://github.com/anicolson/DeepXi) and run a convert script ([convertsnr.py](#convertsnrpy))

---

### extractvfeatures.py
**Short description:** Extract video features from video frames<br>
**Parameters:**

| Parameter Name | Function | Calling from extractfeatures.sh |
|----------------|----------|---------|
| <code>sys.argv[1]</code>, filedir (str) | Source directory of video frame pictures | <code>sdir = $videoframe/$part/Pics</code>
| <code>sys.argv[2]</code>, savedir (str) | Save directory for features | <code>savedir = $videofeature/$part</code> |
| <code>sys.argv[3]</code>, pretrainedmodel (str) | Path to pretrained video model | <code>pretrainedmodeldir = $PRETRAINEDMODEL</code>, usually 'pretrainedvideomodel/Video_only_model.pt', defined in path.sh |
| <code>sys.argv[4]</code>, dset (str) | Dataset part (Train, Test, Val, pretrain) | <code>dset=$part</code> (Test, Train, Val, Pretrain) |
| <code>sys.argv[5]</code>, ifcuda (boolean) | If CUDA should be used | <code>ifcuda=$ifcuda</code> |
| <code>sys.argv[6]</code>, debug (boolean), optional | If debug mode should be used | <code>ifdebug=$ifdebug</code> |

**Explanation:**
- Build a neural network for Lipreading:
  - Input dimension: 256 units, 3-D Convolution Block containing 3 dimensional convolution, batch normalizing, ReLu activation and max pooling
  - Hidden dimension: 512 units, ResNet32 Basic Block
- Initialize the model with the pretrained video model
- Extract features using this network and save them as PyTorch files (.pt) in <code>$savedir/filename</code>

---

### finetune_av.sh
**Short description:**  Code for fine-tuning the audio-visual model<br>
**Parameters:**

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>expdir=$1</code> | Model save directory | <code>$expdiravfine</code>, usually 'exp/fine/AV'|
| <code>dumptraindir=$2</code> | dump training files for av-model | 'dump/avtrain' |
| <code>dumpdecodedir=$3</code> | dump decoding files for av-model |  'dump/avpretraindecode' |
| <code>lmexpdir=$4</code> | path to external language model | <code>$lmexpdir</code> |
| <code>noisetype=$5</code> | Which noise type data is used for decoding, possible noisetype: noise music blur and saltandpepper | <code>$noisetype</code> |
| <code>dict=$6</code> | path to dictionary | <code>$dict</code> |
| <code>bpemodel=$7</code> | path to bpemodel | <code>$bpemodel</code>, usually 'bpemodel=data/lang_char/train_${bpemode}${nbpe}' |
| <code>pretrainedav=$8</code> | path to pretrained av-model | <code>$expdiravpretrain</code> |


**Explanation:**
- fintune Audio-Visual network based on a pretrained model and decode test set afterwards for evaluation

---

### finetune_video.sh
**Short description:**  Code for fine-tuning the video only model<br>
**Parameters:**

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>expdir=$1</code> | Model save directory | <code>$expdirvfine</code>, usually 'exp/fine/V'|
| <code>pretrainexp=$2</code> | path to pretrained video model | <code>$expdirvpretrain</code> |
| <code>dumptraindir=$3</code> | dump training files for video model | 'dump/videotrain' |
| <code>dumpdecodedir=$4</code> | dump decoding files for video model |  'dump/avtraindecode' |
| <code>lmexpdir=$5</code> | path to external language model | <code>$lmexpdir</code> |
| <code>noisetype=$6</code> | Which noise type data is used for decoding, possible noisetype: noise music blur and saltandpepper | <code>$noisetype</code> |
| <code>dict=$7</code> | path to dictionary | <code>$dict</code> |
| <code>bpemodel=$8</code> | path to bpemodel | <code>$bpemodel</code>, usually 'bpemodel=data/lang_char/train_${bpemode}${nbpe}' |


**Explanation:**
- fintune video only network based on a pretrained model and decode test set afterwards for evaluation

---

### LRS3dataprocessing.sh
**Short description:**  Script for preprocess the LRS3 dataset<br>
**Parameters:**

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>videodir=$1</code> | path of LRS3 dataset | $DATALRS3_DIR|
| <code>dset=$2</code> | define set | pretrain |
| <code>ifmulticore=$3</code> | if multi cpu processing, default is true in all scripts | $ifmulticore |
| <code>ifsegpretrain=$4</code> | if we segment pretrain set |  $ifsegment |
| <code>modeltype=$5</code> | extract video features if the modeltype is AV or V | $modeltype |
| <code>ifdebug=$6</code> | with debug, we only use $num utterances from pretrain and $num Utts from Train set | $ifdebug |
| <code>num=$7</code> | number of utterances used from pretrain and Train set | $num |


---

### make_video.py
**Short description:**  Make video ark files<br>
**Parameters:**

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>videodir=$1</code> | source directory | $videofeature |
| <code>dset=$2</code> | savedirectory | data/video |
| <code>ifmulticore=$3</code> | number of parallel processes | $nj |

---

### Openface.sh
**Short description:** Performs OpenFace face recognition and feature extraction for every video file in the respective datasets<br>
**Parameters:**

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>sdir=$1</code> | Directory of the dataset | <code>$DATA_DIR</code>, usually /LRS2, defined in path.sh |
| <code>savedir=$2</code> | Output directory where results (.csv files) should be saved for the given dataset| <code>$facerecog/$part</code>, 'Dataset_processing/Facerecog/$part' |
| <code>dset=$3</code> | Dataset part (Train, Test, Val, pretrain) | <code>$part</code> (Test, Train, Val, Pretrain) |
| <code>OPENFACEDIR=$4</code> | OpenFace build directory | <code>$OPENFACE_DIR</code>, e.g. '/home/foo/AVSR/OpenFace/build/bin' ,defined in path.sh |
| <code>corpus=$5</code> | LRS2 or LRS3 corpus | - |
| <code>nj=$6</code> | number of parallel processes | - |
| <code>ifdebug=$7</code>, default = true | If debug mode should be used | <code>$ifdebug</code> |



**Explanation:**
- For every video (.mp4) file in the LRS2 dataset, face recognition and feature extraction is performed using the OpenFace framework       (https://github.com/TadasBaltrusaitis/OpenFace)
- The features are saved in a seperate .csv file for every video files, whereas the rows of the .csv files corresponds to the video frames and the columns corresponds to the features

---

### Openface_vidaug.sh
**Short description:** Performs OpenFace face recognition and feature extraction for every video file in the respective datasets with blur nose or salt and pepper noise<br>
**Parameters:**

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>sdir=$1</code> | Directory of the dataset | <code>$DATA_DIR</code>, usually /LRS2, defined in path.sh |
| <code>savedir=$2</code> | Output directory where results (.csv files) should be saved for the given dataset| <code>$facerecog/$part</code>, 'Dataset_processing/Facerecog/$part' |
| <code>dset=$3</code> | Dataset part (Train, Test, Val, pretrain) | <code>$part</code> (Test, Train, Val, Pretrain) |
| <code>OPENFACEDIR=$4</code> | OpenFace build directory | <code>$OPENFACE_DIR</code>, e.g. '/home/foo/AVSR/OpenFace/build/bin' ,defined in path.sh |
| <code>corpus=$5</code> | LRS2 or LRS3 corpus | - |
| <code>noisetpye=$6</code> | video noise type blur or salt and pepper noise | 'blur' or 'saltandpepper' |
| <code>nj=$7</code> | number of parallel processes | - |
| <code>ifdebug=$8</code>, default = true | If debug mode should be used | <code>$ifdebug</code> |



**Explanation:**
- For every video (.mp4) file in the LRS2 dataset, face recognition and feature extraction is performed using the OpenFace framework       (https://github.com/TadasBaltrusaitis/OpenFace)
- The features are saved in a seperate .csv file for every video files, whereas the rows of the .csv files corresponds to the video frames and the columns corresponds to the features

---

### prepaudio.py
**Short description:** Build the Kaldi files<br>
**Parameters:**

| Parameter Name | Function | Calling from audio_data_prep.sh |
|----------------|----------|---------|
| <code>sys.argv[1]</code>, sourcedir (str) | The LRS2 dataset dir with subfolders | <code>$sourcedir/main</code>, usually '/LRS2/data/lrs2_v1/mvlrs_v1/main' |
| <code>sys.argv[2]</code>, filelistdir (str) | The directory containing the dataset filelists. The content of the filelists should be like 'it should be like '5535415699068794046/00001' | <code>$metadir</code>, 'data/METADATA' |
| <code>sys.argv[3]</code>, savedir (str) | Save directory, datadir of the clean audio dataset  | <code>$datadir</code>, 'data/audio/clean/$dset' |
| <code>sys.argv[4]</code>, dset (str) | Dataset part (Train, Test, Val, pretrain) | <code>$part</code> (Test, Train, Val, Pretrain) |
| <code>sys.argv[5]</code>, ifmulticore (boolean) | If multi cpu processing should be used | <code>$ifmulticore</code> |


**Explanation:**
- Build the Kaldi files for every dataset
- The resulting kaldi files from this script are text, utt2spk, wav.scp

---

### preppretrainaudio.py
**Short description:** Make the kaldi files for the pretrain set and segment the pretrain audio files<br>
**Parameters:**

| Parameter Name | Function | Calling from audio_data_prep.sh |
|----------------|----------|---------|
| <code>sys.argv[1]</code>, sourcedir (str) | The LRS2 dataset dir for pretrain (e.g. '/LRS2/data/lrs2_v1/mvlrs_v1/pretrain') | <code>$sourcedir/pretrain</code>, usually '/LRS2/data/lrs2_v1/mvlrs_v1/pretrain' |
| <code>sys.argv[2]</code>, filelistdir (str) | The directory containing the dataset filelists. The content of the filelists should be like 'it should be like '5535415699068794046/00001' | <code>$metadir</code>, 'data/METADATA' |
| <code>sys.argv[3]</code>, savedir (str) | Save directory, datadir of the clean audio dataset  | <code>$datadir</code>, 'data/audio/clean/$dset' |
| <code>sys.argv[4]</code>, dset (str) |  Dataset part. For this code dset is pretrain set. | <code>$dset=pretrain</code> |
| <code>sys.argv[5]</code>, savewavdir (str) | Save directory of segmented .wav audio files | <code>Dataset_processing/pretrainsegment</code> |
| <code>sys.argv[6]</code>, ifmulticore (str) | If multi cpu processing should be used | <code>$ifmulticore</code> |
| <code>sys.argv[7]</code>, ifsegment (str) | If do segmentation for pretrain set | <code>$ifsegment</code> |



**Explanation:**
- If segmentation is true, then segment the audio files of the pretrain set into slices of max 5 seconds length
- The resulting kaldi files from this script are text, utt2spk, wav.scp
---

### pretrain.py
**Short description:** Prepare the kaldi files<br>
**Parameters:**

| Parameter Name | Function |
|----------------|----------|
| <code>sys.argv[1]</code>, sourcedir (str) | The LRS2 dataset dir (e.g. /LRS2/data/lrs2_v1/mvlrs_v1/main) |
| <code>sys.argv[2]</code>, filelistdir (str) | The directory containing the dataset Filelists (METADATA) |
| <code>sys.argv[3]</code>, savedir (str), default False | Save directory, datadir of the clean audio dataset |
| <code>sys.argv[4]</code>, dset (str), default False, optional |  Which set. For this code dset is pretrain set | 
| <code>sys.argv[5]</code>, nj (str), default False, optional | Number of multi-processes | 
| <code>sys.argv[6]</code>, segment (str), default False, optional | If do segmentation | 

---

### pretrain_av.sh
**Short description:**  Code for pretraining the audio-visual model<br>
**Parameters:**

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>expdir=$1</code> | Model save directory | <code>$expdiravpretrain</code>, usually 'exp/pretrain/AV'|
| <code>dumptraindir=$2</code> | dump training files for pretrained av model | 'dump/avpretrain' |
| <code>dumpdecodedir=$3</code> | dump decoding files for pretrained av model |  'dump/avpretraindecode' |
| <code>lmexpdir=$4</code> | path to external language model | <code>$lmexpdir</code> |
| <code>noisetype=$5</code> | Which noise type data is used for decoding, possible noisetype: noise music blur and saltandpepper | <code>$noisetype</code> |
| <code>dict=$6</code> | path to dictionary | <code>$dict</code> |
| <code>bpemodel=$7</code> | path to bpemodel | <code>$bpemodel</code>, usually 'bpemodel=data/lang_char/train_${bpemode}${nbpe}' |
| <code>pretrainedaudio=$8</code> | path to pretrained audio model | <code>$expdirapretrain</code>, usually 'exp/pretrain/A' |
| <code>pretrainedvideo=$9</code> | path to pretrained video model | <code>$expdirvpretrain</code>, usually 'exp/pretrain/V' |

**Explanation:**
- pretrain audio-visual model based on pretrained audio and video model

---

### pretrain_video.sh
**Short description:**  Code for pretraining the video only model<br>
**Parameters:**

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>expdir=$1</code> | Model save directory | <code>$expdirvpretrain</code>, usually 'exp/pretrain/V'|
| <code>dumptraindir=$2</code> | dump training files for pretrained video model | 'dump/videopretrain' |
| <code>dumpdecodedir=$3</code> | dump decoding files for pretrained video model |  'dump/avpretraindecode' |
| <code>lmexpdir=$4</code> | path to external language model | <code>$lmexpdir</code> |
| <code>noisetype=$5</code> | Which noise type data is used for decoding, possible noisetype: noise music blur and saltandpepper | <code>$noisetype</code> |
| <code>dict=$6</code> | path to dictionary | <code>$dict</code> |
| <code>bpemodel=$7</code> | path to bpemodel | <code>$bpemodel</code>, usually 'bpemodel=data/lang_char/train_${bpemode}${nbpe}' |


---

### remakeavjson.py
**Short description:** Remake dump files and save them as .json files<br>
**Parameters:**

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>sys.argv[1]</code>, dumpfile (str) | Directory to save dump files | <code>$dumpfile</code>, 'dump/${part}/delta${do_delta}/data_${bpemode}${nbpe}.json' |
| <code>sys.argv[2]</code>, dumpavfile (str) | Directory to save audio-visual dump files | <code>$dumpavfile</code>, 'dumpav/${part}/delta${do_delta}/data_${bpemode}${nbpe}.json' |
| <code>sys.argv[3]</code>, dumpvideofile (str) | Directory to save video dump files | <code>$dumpvideofile</code>, 'dumpvideo/${part}/delta${do_delta}/data_${bpemode}${nbpe}.json' |
| <code>sys.argv[4]</code>, vdir (str) | Video feature directory | <code>$vdir</code>, 'Dataset_processing/Videofeature/$part' |
| <code>sys.argv[5]</code>, snrdir (str) | Directory with SNR saved as .pt files  | <code>$snrdir</code>, 'Dataset_processing/SNRs/$part' |
| <code>sys.argv[6]</code>, vconfdir (str) | Directory with confidence saved as .pt files | <code>$vconfdir</code> 'Dataset_processing/Videodata/$part/Conf' |
| <code>sys.argv[7]</code>, mfccdumpfile (str) | Directory to save mfcc dump files | <code>$mfccdumpfile</code>, 'dumpmfcc/${part}/delta${do_delta}/data_${bpemode}${nbpe}.json' |
| <code>sys.argv[8]</code>, AUdir (str) | Directory with Facial Action Units saved as .pt files | <code>$AUdir</code>, 'Dataset_processing/Videodata/$part/AUs' |
| <code>sys.argv[9]</code>, ifmulticore (boolean) | If multi cpu processing should be used | <code>$ifmulticorejson</code> |

 
---

### remake_dict.py
**Short description:** Replace two dictionaries (char list from input dict to output dict), used for using external language models trained on other text corpora<br>
**Parameters:**

| Parameter Name | Function |
|----------------|----------|
| <code>sys.argv[1]</code>, input_file (str) | Path to input .json file with the source dict |
| <code>sys.argv[2]</code>, output_file (str) | Path to output .json file in which to insert/replace the char list | 

---

### segaugaudio.py

**Parameters:**

| Parameter Name | Function |
|----------------|----------|
| <code>sys.argv[1]</code>, sourcedir (str) | source directory |
| <code>sys.argv[2]</code>, filedir (str) | output file directory | 
| <code>sys.argv[3]</code>, dset (str) |  Which set. For this code dset is pretrain set |
| <code>sys.argv[4]</code>, ifmulticore (str) | If use multi processes |

---

### segmentaudio.py
**Short description:** This code crop the audio file by using the segment information<br>
**Parameters:**

| Parameter Name | Function | Calling from audio_data_prep.sh |
|----------------|----------|---------|
| <code>sys.argv[1]</code>, savedir (str) | Directory in which to save the extracted audio signal | <code>Dataset_processing/pretrainsegment</code> |
| <code>sys.argv[2]</code>, segdir (str) | The dir of the file with the segmentation information path | <code>$datadir</code> |
| <code>sys.argv[3]</code>, ifmulticore (boolean), default False | If multi cpu processing should be used | <code>$ifmulticore</code> |
| <code>sys.argv[4]</code>, debug (boolean), default False, optional | If debug mode should be used | <code>$ifdebug</code> |

---

### segvideo.py
**Short description:** Extract the frames out of the video files (Segment video files) and save the features for all frames in one PyTorch Tensor (.pt file). Extract the features out of the .csv files produced in Stage 3.0 with the OpenFace face recognition (compare [Openface.sh](#openfacesh))<br>

**Parameters:**

| Parameter Name | Function | Calling from extractframs.sh |
|----------------|----------|---------|
| <code>sys.argv[1]</code>, sourcedir (str) | Path to Dataset | <code>$sourcedir</code>, usually '/LRS2/data/lrs2_v1/mvlrs_v1/pretrain' or '/LRS2/data/lrs2_v1/mvlrs_v1/main' |
| <code>sys.argv[2]</code>, savedir (str) | Save directory of the Segmented video data for the given dataset  | <cotest.mdde>$savedir/$dset</code>, 'Dataset_processing/Videodata/$dset' |
| <code>sys.argv[3]</code>, csvdir (str) | The directory of the .csv Files, which contains the Face recognition information extracted by OpenFace for the given dataset | <code>$csvdir/$dset</code>, 'Dataset_processing/Facerecog/$dset' |
| <code>sys.argv[4]</code>, audiodir (str) | The directory in which the clean audio data is saved for the given part | <code>$audiodir/$dset</code>, 'data/audio/clean/$dset' |
| <code>sys.argv[5]</code>, dset (str) | Dataset part (Train, Test, Val, pretrain) | <code>$dset</code>, (Train, Test, Val, pretrain) |
| <code>sys.argv[6]</code>, ifsegment (boolean) | If do segmentation for pretrain set | <code>$ifsegment</code> |
| <code>sys.argv[7]</code>, ifmulticore (boolean) | If multi cpu processing should be used | <code>$ifmulticore</code> |


**Explanation:**
- main Task: Extract the important features out of the .csv files created by OpenFace
- just keep the important features of all OpenFace features: confidence, parts of the facial action units (AUs) and extract pictures of the mouth regions
- For every video (.mp4) file, three .pt files are created which contains the framewise confidence, the Facial Action Units (AUs) and framewise images of the mouth region
- The resulting .pt files are saved in the folders <code>/Dataset_processing/Videodata/$dset/Conf</code> for the confidence features, 
<code>/Dataset_processing/Videodata/$dset/AUs</code> for the Facial Action Units and <code>/Dataset_processing/Videodata/$dset/Pics</code> for the images of the mouth region

- Description of the AU features extracted here (https://www.cs.cmu.edu/~face/facs.htm): 
  - AU12_r: Lip Corner Puller
  - AU15_r: Lip Corner Depressor
  - AU17_r: Chin Raiser
  - AU23_r: Lips Tightener
  - AU25_r: Lips part
  - AU26_r: Jaw Drop

---

### sepaudiovideo.py
**Short description:** This code extract audio files from mp4 files using ffmpeg (Separation of the audio data out of the video files (.mp4)<br>
**Parameters:**

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>sys.argv[1]</code>, savedir (str) | Directory in which to save the audio signal | <code>Dataset_processing/pretrainsegment</code> |
| <code>sys.argv[2]</code>, segdir (str) | The dir of the file with the segmentation information path | <code>$datadir</code> |
| <code>sys.argv[3]</code>, ifmulticore (boolean), default False | If multi cpu processing should be used | <code>$ifmulticore</code> |
| <code>sys.argv[4]</code>, debug (boolean), default False, optional | If debug mode should be used | <code>$ifdebug</code> |

---

### show_result.sh
**Short description:**  Modified Version of the ESPnet show_result.sh script for showing the decoding stats<br>

---

### splitsnr.py
**Short description:**  Split dataset for decoding by their SNR and noise type<br>
**Parameters:**

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>sys.argv[1]</code> | srcdir (str), Source directoy | dump/avtraindecode or dump/avpretraindecode |
| <code>sys.argv[2]</code> | noisecombination(str), Noise combination (noise_None' 'music_None' 'noise_blur' 'noise_saltandpepper) | $noisecombination |
| <code>sys.argv[3]</code> | snrdir (str) |  data/audio/augment/LRS2_decode |

---

### train_audio.sh
**Short description:**  Code for training audio model<br>
**Parameters:**

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>expdir=$1</code> | Model save directory | <code>$expdirapretrain</code>, usually 'exp/pretrain/A'|
| <code>dumptraindir=$2</code> | dump training files for audio model | 'dump/audio' |
| <code>dumpdecodedir=$3</code> | dump decoding files for audio model |  'dump/avpretraindecode' |
| <code>lmexpdir=$4</code> | path to external language model | <code>$lmexpdir</code> |
| <code>noisetype=$5</code> | Which noise type data is used for decoding, possible noisetype: noise music blur and saltandpepper | <code>$noisetype</code> |
| <code>dict=$6</code> | path to dictionary | <code>$dict</code> |
| <code>bpemodel=$7</code> | path to bpemodel | <code>$bpemodel</code>, usually 'bpemodel=data/lang_char/train_${bpemode}${nbpe}' |

---

### videoaug.py
**Short description:**  Create augmented video files with salt and pepper or blurring noise<br>
**Parameters:**

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>sys.argv[1]</code> | filelist (str), Directory to save the file list, which are files augmentated | data/METADATA/Filelist_Test |
| <code>sys.argv[2]</code> | srcdir (str), Directory where save the dataset | $DATA_DIR  |
| <code>sys.argv[3]</code> | savedir (str), Directory to save augmented files |  $videoaug |
| <code>sys.argv[4]</code> | noisetype (str), Video feature directory | 'saltandpepper' or 'blur' |


---

### videodump.py
**Short description:**  Dump video files<br>
**Parameters:**

| Parameter Name | Function | Calling from run.sh |
|----------------|----------|---------|
| <code>sys.argv[1]</code> | dumpfile (str), Directory to save audio-visual dump files | dump/avpretrain or dump/avtrain|
| <code>sys.argv[2]</code> | savedumpdir (str), Directory to save video dump files | dump/videopretrain or dump/videotrain |
| <code>sys.argv[3]</code> | dset (str), Which dataset | <code>$dset</code>, (Train, Test, Val, pretrain) |

dump/avpretrain dump/videopretrain $dset
dump/avtrain dump/videotrain $dset

---
