"""Temporary inference shim for consumers awaiting AQA migration.

Removal tracked in ESPnet PR #6759 (2026-10-15):
https://github.com/espnet/espnet/pull/6759
VERSA consumer migration: https://github.com/wavlab-speech/versa/pull/106
The ten published model cards now use AQA (2026-10-15). VERSA #106 still
needs to merge; see the verification checklist in doc/espnet2_audio_metric.md.
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
