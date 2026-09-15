"""F5-TTS for ESPnet3 (DiT backbone + conditional flow matching).

Self-contained package: the model (``f5tts``, ``cfm``, ``dit``, ``modules``,
``rotary``, ``solvers``, ``utils``), its mel front end (``vocoder_mel``), its
zh+en tokenizer (``pinyin``, ``preprocessor``) and the inference engine
(``inference``). ``f5tts.F5TTS`` is the ESPnet3 model itself, which training
configs reach through ``model._target_`` with ``task:`` left unset.

The model is described in:

Yushen Chen, Zhikang Niu, et al.
"F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching"
https://aclanthology.org/2025.acl-long.313/

If you use this model in your research, please cite the paper above.
"""

#: Default Vocos checkpoint used when ``vocoder_path`` is unset.
VOCOS_DEFAULT_MODEL = "charactr/vocos-mel-24khz"

#: Default BigVGAN checkpoint used when ``vocoder_path`` is unset.
BIGVGAN_DEFAULT_MODEL = "nvidia/bigvgan_v2_24khz_100band_256x"
