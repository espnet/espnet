# LJSpeech Text-to-Speech recipe

Written by Tomoki Hayashi @ Nagoya University (2018/09/09)

## tts1 recipe

`tts1` recipe is based on Tacotron2 [1] (spectrogram prediction network) w/o WaveNet.
Tacotron2 generates log mel-filter bank from text and then converts it to linear spectrogram using inverse mel-basis.
Finally, phase components are recovered with Griffin-Lim.

## tts2 recipe

`tts2` recipe is based on Tacotron2's spectrogram prediction network [1] and Tacotron's CBHG module [2].
Instead of using inverse mel-basis, CBHG module is used to convert log mel-filter bank to linear spectrogram.
The recovery of the phase components is the same as `tts1`.

## Reference

- [1] Shen, Jonathan, et al. "Natural TTS synthesis by conditioning wavenet on mel spectrogram predictions." arXiv preprint arXiv:1712.05884 (2017).
- [2] Wang, Yuxuan, et al. "Tacotron: Towards end-to-end speech synthesis." arXiv preprint arXiv:1703.10135 (2017).
