# CHiME-7 DASR (CHiME-7 Task 1)

### Distant Automatic Speech Transcription with Multiple Devices in Diverse Scenarios

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/chimechallenge.svg?style=social&label=Follow%20%40chimechallenge)](https://twitter.com/chimechallenge)
[![Slack][slack-badge]][slack-invite]

---
This recipe contains the baseline for the CHiME-7 DASR Challenge main track (diarization + ASR).

#### Please refer to `egs2/chime7_task1/README.md` for additional infos (including ranking and evaluation script and dataset generation).

---

## Main Track Baseline (Diarization + ASR)
<img src="https://www.chimechallenge.org/current/task1/images/baseline.png" width="450" height="120" />

The baseline system for the main track is composed of three main modules (with an optional channel selection module):

0. **Diarizer**: the diarizer module performs multi-channel diarization and is used to obtain utterance-wise segmentation
for each speaker. <br>
Here we use a system based on [Pyannote diarization pipeline](https://huggingface.co/pyannote/speaker-diarization) [1], [2] modified
to handle multiple channels.

After diarization **we re-use the same components as used in the acoustic robustness sub-track** (where instead oracle diarization is used):

1. **Guided Source Separation** (GSS) [3], here we employ the GPU-based version (much faster) from [Desh Raj](https://github.com/desh2608/gss).
2. End-to-end **ASR** model based on [4], which is a transformer encoder/decoder model trained <br>
with joint CTC/attention [5]. It uses WavLM [6] as a feature extractor.
3. Optional **Automatic Channel selection** based on Envelope Variance Measure (EV) [7].


### Pyannote-based Diarization System

As said this diarization system is based on [Pyannote diarization pipeline](https://huggingface.co/pyannote/speaker-diarization) as described in
this [technical report](https://huggingface.co/pyannote/speaker-diarization/resolve/main/technical_report_2.1.pdf). <br>
It is a very effective diarization system that uses a local EEND-based segmentation model [1] and leverages this model
for extracting overlap-aware embeddings.

Here we use the pre-trained [Pyannote diarization pipeline](https://huggingface.co/pyannote/speaker-diarization) and fine-tune only the
segmentation model [Pyannote segmentation model](https://huggingface.co/pyannote/segmentation) on CHiME-6 training data (using Mixer 6 speech dev for validation). <br>

We found that fine-tuning using the CHiME-6 manual annotation was better than using the CHiME-6 forced-alignment annotation,
which is made available, also for the training set [here](https://github.com/chimechallenge/CHiME7_DASR_falign). <br>

The fine-tuned segmentation model is available in HuggingFace: [popcornell/pyannote-segmentation-chime6-mixer6](https://huggingface.co/popcornell/pyannote-segmentation-chime6-mixer6)

#### Extension to multiple channels
Here, since multiple channels are available, we run the segmentation model on all the channels
and then select the best channel based on the output of the segmentation model (the channel with the most speech activity is chosen).
<br> Then only the best channel is used for diarization. Note that this happens for each different 5 second chunk.

**This very crude selection policy assumes that the segmentation model is very robust against noisy channels.** <br>
It also does not account for the fact that one speaker could be prevalent in one channel while another in another channel. <br>
It was mainly chosen due to the fact that it keeps the inference very fast (opposed to e.g. using all channels or top-k in clustering). <br>
There is then a lot to improve upon.

#### Results

We report the results with this Pyannote-based multi-channel pipeline hereafter (as obtained with the challenge evaluation script ( `egs2/chime7_task1/README.md`))

```bash
###################################################
### Metrics for all Scenarios ###
###################################################
+----+------------+---------------+---------------+----------------------+----------------------+---------------+-----------+----------+-------------+--------------------+--------------------------+-----------------+-----------------+----------------------+--------+-----------------+-------------+--------------+----------+
|    | scenario   |   num spk hyp |   num spk ref |   tot utterances hyp |   tot utterances ref |   false alarm |   correct |    total |   confusion |   missed detection |   diarization error rate |   speaker error |   speaker count |   Jaccard error rate |   hits |   substitutions |   deletions |   insertions |      wer |
|----+------------+---------------+---------------+----------------------+----------------------+---------------+-----------+----------+-------------+--------------------+--------------------------+-----------------+-----------------+----------------------+--------+-----------------+-------------+--------------+----------|
|  0 | chime6     |             8 |             8 |                 3178 |                 6643 |        407.34 |   8197.73 | 13528.8  |     1852.16 |            3478.88 |                 0.424161 |         4.22922 |               8 |             0.528653 |  24803 |           12649 |       21429 |         4156 | 0.649344 |
|  0 | dipco      |            20 |            20 |                 2129 |                 3673 |        356.26 |   5675.47 |  7753.04 |      990.9  |            1086.67 |                 0.313919 |         8.6079  |              20 |             0.430395 |  15374 |            7409 |        7183 |         3542 | 0.605153 |
|  0 | mixer6     |           141 |           118 |                 9045 |                14803 |        446.65 |  36896    | 44557.9  |      729.59 |            6932.28 |                 0.181977 |        28.5003  |             118 |             0.241528 | 115698 |           17401 |       15882 |         8471 | 0.280264 |
+----+------------+---------------+---------------+----------------------+----------------------+---------------+-----------+----------+-------------+--------------------+--------------------------+-----------------+-----------------+----------------------+--------+-----------------+-------------+--------------+----------+
####################################################################
### Macro-Averaged Metrics across all Scenarios (Ranking Metric) ###
####################################################################
+----+---------------+---------------+---------------+----------------------+----------------------+---------------+-----------+---------+-------------+--------------------+--------------------------+-----------------+-----------------+----------------------+---------+-----------------+-------------+--------------+----------+
|    | scenario      |   num spk hyp |   num spk ref |   tot utterances hyp |   tot utterances ref |   false alarm |   correct |   total |   confusion |   missed detection |   diarization error rate |   speaker error |   speaker count |   Jaccard error rate |    hits |   substitutions |   deletions |   insertions |      wer |
|----+---------------+---------------+---------------+----------------------+----------------------+---------------+-----------+---------+-------------+--------------------+--------------------------+-----------------+-----------------+----------------------+---------+-----------------+-------------+--------------+----------|
|  0 | macro-average |       56.3333 |       48.6667 |                 4784 |                 8373 |       403.417 |   16923.1 | 21946.6 |     1190.88 |            3832.61 |                 0.306686 |         13.7791 |         48.6667 |             0.400192 | 51958.3 |         12486.3 |     14831.3 |      5389.67 | 0.511587 |
+----+---------------+---------------+---------------+----------------------+----------------------+---------------+-----------+---------+-------------+--------------------+--------------------------+-----------------+-----------------+----------------------+---------+-----------------+-------------+--------------+----------+
```
We can see that the DER and JER values are quite competitive on CHiME-6 with the state-of-the-art (e.g. JHU CHiME-6 submission) but
the WER figures are quite poor. <br>
**Note that here we report DER and JER which are computed against manual annotation for CHiME-6.** <br>
The DER and JER obtained with respect to the [forced-alignment annotation](https://github.com/chimechallenge/CHiME7_DASR_falign) will be higher. In general this model can be optimized to reach ~46% DER with respect to such annotation on CHiME-6 (thus roughly comparable to Kaldi TS-VAD implementation) but
we found this configuration to lead to worse WER overall.

This may due to the fact that we use an E2E ASR system which may be more sensitive to segmentation errors compared to
hybrid ASR models. We found that using an higher weight in decoding for CTC helped a bit (we use here 0.6, see `conf/decode_asr_transformer.yaml`).

It is worth to point out also that it is quite challenging to optimize the diarization
hyperparameters (for example merging the segments that are X apart) for all three scenarios. <br> E.g. best parameters for CHiME-6 lead to degradation to
Mixer 6 performance. <br>



### Reproducing the Baseline Results.

#### Training the ASR model

Follow the instructions in the acoustic sub-track recipe `egs2/chime7_task1/README.md`.
Then you can create a symbolic link to your trained ASR model e.g. if it is in the folder (using the standard config in the acoustic robustness sub-track) <br>
`exp/MY_ASR_SYSTEM$
`

Then you can create a symbolic link to this ASR model in the exp folder here:
```bash
mkdir exp # if it does not exists yet in this folder it should be created
ln -s ../asr1/exp/MY_ASR_SYSTEM ./exp
```

**Important** <br>
The recipe stage 0 here will re-create the data. If you want to avoid
this and use the data already created in the acoustic sub-track recipe `egs2/chime7_task1/README.md`
you can also create a symbolic link to that one:
```bash
ln -s ../asr1/chime7_task1 .
```

#### Main Track with Pyannote-based Diarization System
To reproduce our results which use our pre-trained ASR model [https://huggingface.co/popcornell/chime7_task1_asr1_baseline](https://huggingface.co/popcornell/chime7_task1_asr1_baseline) and pre-trained
[pyannote segmentation model](https://huggingface.co/popcornell/pyannote-segmentation-chime6-mixer6)
you can run:
```bash
./run.sh --chime7-root YOUR_PATH_TO_CHiME7_ROOT --stage 1 --ngpu YOUR_NUMBER_OF_GPUs \
--use-pretrained popcornell/chime7_task1_asr1_baseline \
--decode-only 1 --gss-max-batch-dur 30-360-DEPENDING_ON_GPU_MEM \
--pyan-use-pretrained popcornell/pyannote-segmentation-chime6-mixer6
```
You can also play with diarization hyperparameters such as:
1. `--diar-merge-closer`: merge segments after diarization from same speaker that are closer than this value.
2. `--diar-max-length-merged`: max length allowed for segments that are merged (default 60, you may want to lower this due to GSS memory consumption).
3. `--pyannote-max_batch_size`: max batch size used in inference when extracting embeddings (increase to speed up, default 32).
---
If you want to run the recipe from scratch, **including dataset generation** and pyannote segmentation
model finetuning you can run it from stage 0:
```bash
./run.sh --chime6-root YOUR_PATH_TO_CHiME6 --dipco-root PATH_WHERE_DOWNLOAD_DIPCO \
--mixer6-root YOUR_PATH_TO_MIXER6 --stage 0 --ngpu YOUR_NUMBER_OF_GPUs \
--use-pretrained popcornell/chime7_task1_asr1_baseline \
--decode-only 1 --gss-max-batch-dur 30-360-DEPENDING_ON_GPU_MEM \
--pyan-use-pretrained popcornell/pyannote-segmentation-chime6-mixer6
```
---
**If you want only to generate data you can run only stage 0.**
```bash
./run.sh --chime6-root YOUR_PATH_TO_CHiME6 --dipco-root PATH_WHERE_DOWNLOAD_DIPCO \
--mixer6-root YOUR_PATH_TO_MIXER6 --stage 0 --stop-stage 0
```
Please refer to `egs2/chime7_task1/README.md` for additional infos such as
generating evaluation data when this will be made available.
---

In the optional stage 1 the pyannote segmentation model is fine-tuned. You can run only this stage using to obtain
your own fine-tuned segmentation model.
```bash
./run.sh --chime7-root YOUR_PATH_TO_CHiME7_ROOT --stage 1 --stop-stage 1 --ngpu YOUR_NUMBER_OF_GPUs
```
You can also play with diarization hyperparameters such as:
1. `--pyan-learning-rate`: learning rate to use in fine-tuning.
2. `--pyan-batch-size`: batch size to use in fine-tuning.

You can check the fine-tuning progress by using tensorboard e.g.:
```bash
tensorboard --logdir=./exp/pyannote_finetune/lightning_logs/
```

## References

[1] Bredin, Hervé, and Antoine Laurent. "End-to-end speaker segmentation for overlap-aware resegmentation." arXiv preprint arXiv:2104.04045 (2021). <br>
[2] Bredin, Hervé, et al. "Pyannote. audio: neural building blocks for speaker diarization." ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020.  <br>
[3] Boeddeker, C., Heitkaemper, J., Schmalenstroeer, J., Drude, L., Heymann, J., & Haeb-Umbach, R. (2018, September). Front-end processing for the CHiME-5 dinner party scenario. In CHiME5 Workshop, Hyderabad, India (Vol. 1). <br>
[4] Chang, X., Maekaku, T., Fujita, Y., & Watanabe, S. (2022). End-to-end integration of speech recognition, speech enhancement, and self-supervised learning representation. <https://arxiv.org/abs/2204.00540> <br>
[5] Kim, S., Hori, T., & Watanabe, S. (2017, March). Joint CTC-attention based end-to-end speech recognition using multi-task learning. Proc. of ICASSP (pp. 4835-4839). IEEE. <br>
[5] Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., Chen, Z., ... & Wei, F. (2022). Wavlm: Large-scale self-supervised pre-training for full stack speech processing. IEEE Journal of Selected Topics in Signal Processing, 16(6), 1505-1518. <br>
[7] Wolf, M., & Nadeu, C. (2014). Channel selection measures for multi-microphone speech recognition. Speech Communication, 57, 170-180.  <br>


[slack-badge]: https://img.shields.io/badge/slack-chat-green.svg?logo=slack
[slack-invite]: https://join.slack.com/t/chime-fey5388/shared_invite/zt-1oha0gedv-JEUr1mSztR7~iK9AxM4HOA
[twitter]: https://twitter.com/chimechallenge<h2>References</h2>
