"""Speaker system base entrypoints."""

from espnet3.systems.spk.base.embed_extract import (
    extract_embed,
    get_embed_extract_parser,
    main_embed_extract,
)
from espnet3.systems.spk.base.inference import (
    Speech2Embedding,
    get_inference_parser,
    infer,
    main_inference,
)
from espnet3.systems.spk.base.training import get_train_parser, main_train, train

__all__ = [
    "Speech2Embedding",
    "extract_embed",
    "get_embed_extract_parser",
    "get_inference_parser",
    "get_train_parser",
    "infer",
    "main_embed_extract",
    "main_inference",
    "main_train",
    "train",
]
