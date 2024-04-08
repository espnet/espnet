# LJSpeech Text-to-Speech recipe

Written by Tomoki Hayashi @ Nagoya University (2018/09/09)

## tts1 recipe

`tts1` recipe is based on Tacotron2 [1] (spectrogram prediction network) w/o WaveNet.
Tacotron2 generates log mel-filter bank from text and then converts it to linear spectrogram using inverse mel-basis.
Finally, phase components are recovered with Griffin-Lim.

- (2019/06/16) we also support TTS-Transformer [3].
- (2019/06/17) we also support Feed-forward Transformer [4].
- (2019/12/12) we also support Knowledge distillation Feed-forward Transformer [4].

You can view the several configurations `conf/tuning/*.yaml` and you can switch them via `--train-config` in `run.sh`.
There is a brief explanation of the configuration in the header of each yaml file.
Please check it to understand the difference of each configuration.

If you want to train fastspeech, at first, you need to train Tacotron 2 or Transformer.
Note that both Tactoron 2 and Transformer can be used as a teacher of FastSpeech.

After that you can train two types of FastSpeech:

1. FastSpeech without Knowledge distillation (`v1`, `v2`, `v3`)
2. FastSpeech with Knowledge distillation (`v4`)

In the case (1), you need to update `teacher-model` in yaml config to use your trained Tacotron 2 or Transformer.
In this case, groundtruth of mel spectrogram and duration calculated by the teacher model with teacher-forcing are used as targets of FastSpeech.
Note that the duration is calculated on-the-fly during training.

In the case (2), you need to specify `teacher_model_path` in `run.sh`.
If this case, first we generate training data (mel-spectrogram and durations) without teacher-forcing using the teacher model and dump them.
Note that to generate training data we use special decoding config `conf/decode_for_knowledge_dist.yaml` and we do not use groundtruth of mel-spectrogram for training.
After that we re-generate the json files for FastSpeech training and then train FastSpeech with new json files.

Since we use generated mel-spectrogram as a target, we need to carefully check the quality, especially in the case of Transformer (In our experiments, Tacotron 2 with attention constraint is more stable than Transformer).
To address this issue, we prepare the filtering method using focus rate [4] to remove bad generated samples.
But you need to carefully check the threshold of the focus rate to prepare the good training data.

## tts2 recipe

`tts2` recipe is based on Tacotron2's spectrogram prediction network [1] and Tacotron's CBHG module [2].
Instead of using inverse mel-basis, CBHG module is used to convert log mel-filter bank to linear spectrogram.
The recovery of the phase components is the same as `tts1`.

## Reference

- [1] Shen, Jonathan, et al. "Natural TTS synthesis by conditioning wavenet on mel spectrogram predictions." arXiv preprint arXiv:1712.05884 (2017).
- [2] Wang, Yuxuan, et al. "Tacotron: Towards end-to-end speech synthesis." arXiv preprint arXiv:1703.10135 (2017).
- [3] Li, Naihan, et al. "Close to human quality TTS with transformer." arXiv preprint arXiv:1809.08895 (2018).
- [4] Ren, Yi, et al. "FastSpeech: Fast, Robust and Controllable Text to Speech." arXiv preprint arXiv:1905.09263 (2019).
