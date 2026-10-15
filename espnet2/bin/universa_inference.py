"""Temporary inference shim for consumers awaiting AQA migration.

Removal tracked in ESPnet PR #6759 (2026-10-15):
https://github.com/espnet/espnet/pull/6759
VERSA consumer migration: https://github.com/wavlab-speech/versa/pull/106
The published model cards and linked Colab still advertise UniversaInference;
see the remaining consumer checklist in doc/espnet2_audio_metric.md.
"""

from espnet2.bin.aqa_inference import (  # noqa: F401
    AqaInference,
    get_parser,
    inference,
    main,
)

UniversaInference = AqaInference

if __name__ == "__main__":
    main()
