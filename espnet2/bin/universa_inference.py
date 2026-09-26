"""Legacy entry point; new recipes use audio_metric_inference."""

from espnet2.bin.audio_metric_inference import (  # noqa: F401
    UniversaInference,
    get_parser,
    inference,
    main,
)

if __name__ == "__main__":
    main()
