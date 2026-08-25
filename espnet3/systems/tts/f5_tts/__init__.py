"""F5-TTS for ESPnet3 (DiT backbone + conditional flow matching).

Self-contained package: the model (``f5tts``, ``cfm``, ``backbones.dit``,
``modules``, ``rotary``, ``solvers``, ``utils``), its mel front end
(``vocoder_mel``), its zh+en tokenizer (``pinyin``, ``preprocessor``), its
learning-rate schedule (``scheduler``), the inference engine (``inference``),
and ``builder.build_f5_tts_model``, which configs reach through ``_target_``.

Nothing is re-exported here on purpose: several submodules pull heavy optional
dependencies, so importing the package must stay cheap. Use full dotted paths.
"""
