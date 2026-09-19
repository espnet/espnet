"""Fetch a published model for the ``from_pretrained`` constructors."""

import logging
from typing import Dict


def download_pretrained(model_tag: str) -> Dict:
    """Return the constructor keyword arguments for a published model.

    Wraps ``espnet_model_zoo`` so every inference class gives the same
    message when the package is missing, and so a bundle produced by
    espnet3's ``pack_model`` is recognised: its ``meta.yaml`` names
    ``inference_config`` / ``training_config`` rather than the
    ``<task>_train_config`` / ``<task>_model_file`` pairs these classes take,
    and it has to be loaded through espnet3.
    """
    try:
        from espnet_model_zoo.downloader import ModelDownloader
    except ImportError:
        logging.error(
            "`espnet_model_zoo` is not installed. "
            "Please install via `pip install -U espnet_model_zoo`."
        )
        raise
    kwargs = ModelDownloader().download_and_unpack(model_tag)
    if "inference_config" in kwargs or "training_config" in kwargs:
        raise RuntimeError(
            f"{model_tag} was published with espnet3's pack_model; load it with "
            "espnet3.publication.inference_model.InferenceModel.from_pretrained("
            f"{model_tag!r}) instead (trust_user_code=True if the bundle ships "
            "its own code)."
        )
    return kwargs
