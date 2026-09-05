# VCTK 0.92 data preparation

This recipe supports both common VCTK layouts:

- VCTK 0.92: `wav48_silence_trimmed/*/*_mic1.flac`
- Legacy VCTK: `wav48/*/*.wav`

The layout is detected automatically. The microphone option applies to VCTK
0.92; legacy `wav48` is already a single recording per utterance. The recipe
does not modify or call the VCTK ASR/TTS local data scripts and does not require
external HTS labels.
