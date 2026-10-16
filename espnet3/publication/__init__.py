"""Publication-facing APIs for packaged ESPnet3 models."""

from espnet3.publication.schema import PACK_SCHEMA_VERSION
from espnet3.publication.inference_model import InferenceModel

__all__ = ["PACK_SCHEMA_VERSION", "InferenceModel"]
