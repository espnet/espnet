# CHiME-7 DASR (CHiME-7 Task 1)

### Distant Automatic Speech Transcription with Multiple Devices in Diverse Scenarios

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/chimechallenge.svg?style=social&label=Follow%20%40chimechallenge)](https://twitter.com/chimechallenge)
[![Slack][slack-badge]][slack-invite]

---
This recipe contains the baseline for the CHiME-7 DASR Challenge main track (diarization + ASR).

### Please refer to `egs2/chime7_task1/README.md` for additional infos (including ranking and evaluation script and dataset generation).

---

## <a id="installation">2. Installation </a>

Follow the instructions in `egs2/chime7_task1/README.md` about how to install ESPNEt2 and the required
dependencies for this recipe.


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

We report the results with this Pyannote-based multi-channel pipeline hereafter (as obtained with the challenge evaluation script ( `egs2/chime7_task1/README.md`)). <br>
More detailed results for each scenario (as produced by the evaluation script in `local/da_wer_scoring.py`) are in `baseline_logs/inference.log` here.

```bash
###################################################
### Metrics for all Scenarios ###
###################################################
+----+------------+---------------+---------------+----------------------+----------------------+--------------------+----------+-----------+-------------+---------------+--------------------------+-----------------+-----------------+----------------------+--------+-----------------+-------------+--------------+----------+
|    | scenario   |   num spk hyp |   num spk ref |   tot utterances hyp |   tot utterances ref |   missed detection |    total |   correct |   confusion |   false alarm |   diarization error rate |   speaker error |   speaker count |   Jaccard error rate |   hits |   substitutions |   deletions |   insertions |      wer |
|----+------------+---------------+---------------+----------------------+----------------------+--------------------+----------+-----------+-------------+---------------+--------------------------+-----------------+-----------------+----------------------+--------+-----------------+-------------+--------------+----------|
|  0 | chime6     |             8 |             8 |                 3321 |                 6643 |            3021.8  | 13528.8  |   8544.78 |     1962.19 |        424.44 |                 0.399772 |         4.09521 |               8 |             0.511901 |  27149 |           11830 |       19902 |         5013 | 0.624055 |
|  0 | dipco      |            20 |            20 |                 2251 |                 3673 |             939.43 |  7753.04 |   5804.02 |     1009.59 |        365.67 |                 0.298553 |         8.28111 |              20 |             0.414056 |  16614 |            7137 |        6215 |         3622 | 0.566442 |
|  0 | mixer6     |           141 |           118 |                 9807 |                14803 |            6169.91 | 44557.9  |  37631.8  |      756.2  |        455.37 |                 0.165661 |        26.9167  |             118 |             0.228108 | 122549 |           14000 |       12432 |         7210 | 0.225814 |
+----+------------+---------------+---------------+----------------------+----------------------+--------------------+----------+-----------+-------------+---------------+--------------------------+-----------------+-----------------+----------------------+--------+-----------------+-------------+--------------+----------+
####################################################################
### Macro-Averaged Metrics across all Scenarios (Ranking Metric) ###
####################################################################
+----+---------------+---------------+---------------+----------------------+----------------------+--------------------+---------+-----------+-------------+---------------+--------------------------+-----------------+-----------------+----------------------+---------+-----------------+-------------+--------------+----------+
|    | scenario      |   num spk hyp |   num spk ref |   tot utterances hyp |   tot utterances ref |   missed detection |   total |   correct |   confusion |   false alarm |   diarization error rate |   speaker error |   speaker count |   Jaccard error rate |    hits |   substitutions |   deletions |   insertions |      wer |
|----+---------------+---------------+---------------+----------------------+----------------------+--------------------+---------+-----------+-------------+---------------+--------------------------+-----------------+-----------------+----------------------+---------+-----------------+-------------+--------------+----------|
|  0 | macro-average |       56.3333 |       48.6667 |              5126.33 |                 8373 |            3377.05 | 21946.6 |   17326.9 |     1242.66 |        415.16 |                 0.287995 |         13.0977 |         48.6667 |             0.384688 | 55437.3 |           10989 |     12849.7 |      5281.67 | 0.472104 |
+----+---------------+---------------+---------------+----------------------+----------------------+--------------------+---------+-----------+-------------+---------------+--------------------------+-----------------+-----------------+----------------------+---------+-----------------+-------------+--------------+----------+
```

We can see that the DER and JER values are quite competitive on CHiME-6 with the state-of-the-art but
the WER figures are quite poor. <br>
**Note that here we report DER and JER which are computed against manual annotation with 0.25 seconds collar for CHiME-6.** <br>
The DER and JER obtained with respect to the [forced-alignment annotation](https://github.com/chimechallenge/CHiME7_DASR_falign) are a bit lower actually (38.94%). <br>
Note that the previous challenge used no DER and JER collars e.g. the [Kaldi TS-VAD implementation](https://github.com/kaldi-asr/kaldi/tree/master/egs/chime6/s5c_track2) obtains
around 44% DER with respect to forced alignment annotation with no collar. <br>
This same model achieves around 54% DER w.r.t. the same annotation but it is because it is optimized towards looser segmentation which
we found it yielded better WER especially on Mixer 6 compared to optimizing the pipeline towards lower DER w.r.t. forced alignment no collar ground truth.


The high WER figures may due to the fact that we use an E2E ASR system which may be more sensitive to segmentation errors compared to
hybrid ASR models. We found that using an higher weight in decoding for CTC helped a bit (we use here 0.6, see `conf/decode_asr_transformer.yaml`). <br>
On the other hand, using ESPNet2 E2E ASR allows to keep the baseline arguably simpler (than e.g. Kaldi) and allows more easily to explore some techniques such as serialized output training,
target speaker ASR and E2E integration with speech enhancement and separation front-ends. <br>
But you are free to explore other techniques (e.g. by using [K2](https://github.com/k2-fsa/k2), which is also integrated with
[lhotse](https://github.com/lhotse-speech/lhotse), used in this recipe for data preparation).

It is worth to point out also that it is quite challenging to optimize the diarization
hyper-parameters (for example merging the segments that are X apart) for all three scenarios. <br> E.g. best parameters for CHiME-6 lead to degradation to
Mixer 6 performance. <br>



### Reproducing the Baseline Results.

#### Training the ASR model

Follow the instructions in the acoustic sub-track recipe `egs2/chime7_task1/README.md`. <br>
Then you can create a symbolic link to your trained ASR model (which will be in the `../asr1/exp` folder). <br>
For example, to create a symbolic link to your trained ASR model in the exp folder here in `diar_asr1`:
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
[pyannote segmentation model](https://huggingface.co/popcornell/pyannote-segmentation-chime6-mixer6), on the dev set,
you can run:
```bash
./run.sh --chime7-root YOUR_PATH_TO_CHiME7_ROOT --stage 2 --ngpu YOUR_NUMBER_OF_GPUs \
--use-pretrained popcornell/chime7_task1_asr1_baseline \
--decode-only dev --gss-max-batch-dur 30-360-DEPENDING_ON_GPU_MEM \
--pyan-use-pretrained popcornell/pyannote-segmentation-chime6-mixer6
```
You can also play with diarization hyperparameters such as:
1. `--diar-merge-closer`: merge segments after diarization from same speaker that are closer than this value.
2. `--diar-max-length-merged`: max length allowed for segments that are merged (default 60, you may want to lower this due to GSS memory consumption).
3. `--pyannote-max_batch_size`: max batch size used in inference when extracting embeddings (increase to speed up, default 32).

as said merge-closer can have quite an impact on the final WER.

**NOTE**
We found the diarization baseline to be highly sensitive to the `diar-merge-closer` parameter and
to the CUDA/CUDNN version used. <br>
For example, the best results on our side were obtained with `conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia`.
This however was by using Ampere devices (A100) on our side, and the results might
change for you if your machine is different. <br>
See [this Pyannote issue](https://github.com/pyannote/pyannote-audio/issues/1370) related to replicability of the diarization baseline, where we have
reported the full specs of our system and the conda environment used. <br>

To enhance replicability, we provide in this [repository](https://github.com/popcornell/CHiME7DASRDiarizationBaselineJSONs) our pre-computed outputs
for the diarization baseline.
You can use them in this recipe by passing `--download-baseline-diarization 1 ` this
will skip your "local" diarization baseline and instead download directly our predictions.

---
If you want to run this recipe from scratch, **including dataset generation** and pyannote segmentation
model fine-tuning you can run it from stage 0 (use `--decode-only eval` for evaluation set):
```bash
./run.sh --chime6-root YOUR_PATH_TO_CHiME6 --dipco-root PATH_WHERE_DOWNLOAD_DIPCO \
--mixer6-root YOUR_PATH_TO_MIXER6 --stage 0 --ngpu YOUR_NUMBER_OF_GPUs \
--use-pretrained popcornell/chime7_task1_asr1_baseline \
--decode-only dev --gss-max-batch-dur 30-360-DEPENDING_ON_GPU_MEM \
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

## Common Issues
1. ```huggingface_hub.utils._validators.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': 'exp/pyannote_finetuned/lightning_logs/version_0/checkpoints/best.ckpt'. Use `repo_type` argument if needed.
It requests “.../version_0/….” while I have “.../version_461/…``` the pyannote diarization pipeline uses **pytorch-lightning** which puts the experiment logs and checkpoint in `lightning_logs/version_XX` starting from 0
and increasing at each new launch of training. Probably the first attempts failed on your end and the checkpoint was put into the 461th folder. You should delete the `lightning_logs` folder and restart the recipe or copy the checkpoint in the `version_0` folder as
it's there where the `local/pyannote_diarize.py` script expects the model to be.

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
