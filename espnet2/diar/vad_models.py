"""The published VAD model of each mode and its decision rule.

Changing the VAD model of ESPnet means editing this file: espnet2.bin.diar_inference,
`espnet detect`, the MCP tool and the Space (egs2/frontia_vad_v1/diar1/demo) read
their defaults from here. The Space's model card lists the
tags as well, because Hugging Face reads them from its front matter;
test/espnet2/bin/test_demo_apps.py checks that it agrees.

This module imports nothing, so `espnet --help` can read it without loading
torch.
"""

# The published model of each mode.
# Both modes use the dual-mode TCN of v2.3:
# - online: run with look-ahead 0, so it reads no future frame and streams;
# - offline: run with its largest look-ahead (124 frames, 0.99 s), which places
#   the boundaries better when the whole file is there.
# A local .zip from stage 8 can stand in for either.
VAD_MODELS = {
    "online": "espnet/frontia_vad_v2_3",
    "offline": "espnet/frontia_vad_v2_3",
}

# Each mode's decision rule, make_rttm.py's as stage 7 scores it: a frame is
# speech when its probability exceeds the threshold, then a centred median
# filter of that many frames smooths the 0/1 sequence, as (threshold, median).
# Both can be changed: VoiceActivityDetector(threshold=, median=), and
# `espnet detect --threshold --median`.
#
# The centred median of M frames reads (M - 1) / 2 frames after each frame, so
# it delays every decision by (M - 1) / 2 x 8 ms: 96 ms at 25. The default
# online rule is therefore NOT strictly causal, although the model reads no
# future audio; median 1 gives a strictly causal decision with no delay.
VAD_RULES = {
    # v2.3 at look-ahead 0: the online rule of its model card, tuned on
    # ami_mh_dev_v1. Decisions come 96 ms after the audio.
    "online": (0.6, 25),
    # v2.3 at look-ahead 124 (992 ms) and the same median, so decisions come
    # 1088 ms after the audio. The model card gives median 1; on the 23
    # benchmark sets median 25 scores better (macro DetER 9.79 against 9.90,
    # macro frame EER 12.37 against 12.49, threshold 0.6 for both) and removes
    # the one-frame segments median 1 leaves at speech edges.
    "offline": (0.6, 25),
}
