"""Compatibility entry point; new recipes use aqa_inference."""

from espnet2.bin.aqa_inference import (  # noqa: F401
    AqaInference,
    get_parser,
    inference,
    main,
)

UniversaInference = AqaInference

if __name__ == "__main__":
    main()
