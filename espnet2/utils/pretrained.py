"""Fetch a published model for the ``from_pretrained`` constructors."""

import inspect
import logging
import re
from typing import Any, Dict


class ModelTagError(RuntimeError):
    """The tag names a model that this inference class cannot load.

    Its own type because both front ends - ``espnet2.bin.cli`` and
    ``espnet.load`` - report it as a choice the user can correct rather than
    as a traceback.
    """


def download_pretrained(model_tag: str) -> Dict:
    """Return the constructor keyword arguments for a published model.

    Wraps ``espnet_model_zoo`` so every inference class gives the same
    message when the package is missing, and so a bundle produced by
    espnet3's ``pack_model`` is recognised: its ``meta.yaml`` names
    ``inference_config`` / ``training_config`` rather than the
    ``<task>_train_config`` / ``<task>_model_file`` pairs these classes take,
    and it has to be loaded through espnet3.
    """
    _refuse_if_speechlm(model_tag)

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


def _refuse_if_speechlm(model_tag: str) -> None:
    """Send a SpeechLM release to the loader that can read it.

    These are published as loose files - a train config, a ``.pt`` holding
    ``{"module": state_dict}``, decoding configs - rather than as an
    espnet_model_zoo pack, so the downloader below cannot unpack one and
    none of these classes can be built from what it would return. Worse,
    ``espnet/bagpiper-tts-sft`` carries ``text-to-speech`` on the Hub, so
    ``espnet.load`` infers ``tts`` for it and the failure arrives as a
    puzzle about Text2Speech. Say so before anything is fetched.
    """
    from espnet2.bin.speechlm_inference import RELEASES

    if str(model_tag) not in RELEASES:
        return
    raise ModelTagError(
        f"{model_tag} is a SpeechLM release ("
        f"{RELEASES[str(model_tag)]['what']}), which is published as loose "
        "files rather than as an espnet_model_zoo pack. Load it with "
        "espnet2.bin.speechlm_inference.from_pretrained("
        f"{str(model_tag)!r}), or address a server with "
        "espnet2.bin.speechlm_inference.from_server()."
    )


def build_pretrained(
    loader: Any,
    model_tag: str,
    device: str,
    what: str,
    remedy: str,
    **kwargs: Any,
):
    """Load a published model, or say why this tag cannot serve this task.

    The downloader hands a model's own artifact keys to the constructor, so a
    tag published for another task arrives as an unexpected keyword argument.
    Only that is translated: every other TypeError is a bug worth seeing in
    full rather than being blamed on the user's choice of model.

    Args:
        loader: The inference class, e.g. ``Text2Speech``.
        model_tag: The tag to load.
        device: Passed to the constructor.
        what: How the caller names itself in the message, e.g.
            ``"`espnet synthesize`"`` or ``"espnet.load(task='tts')"``.
        remedy: One sentence telling the user how to pick another tag.
        **kwargs: Further constructor arguments.
    """
    try:
        return loader.from_pretrained(model_tag, device=device, **kwargs)
    except TypeError as e:
        unexpected = re.search(r"unexpected keyword argument '([^']+)'", str(e))
        if not unexpected:
            raise
        name = unexpected.group(1)
        if name in inspect.signature(loader.__init__).parameters:
            raise  # the constructor does take it; something else went wrong
        if name in kwargs:
            # The caller named it themselves, so the published model cannot be
            # blamed for it: `from_pretrained` merges the two sets of keyword
            # arguments and the message does not say which one it came from.
            # The real TypeError stands, and it names the key.
            raise
        raise ModelTagError(
            f"{model_tag} does not look like a model for {what}: it "
            f"was published with {name}, which {loader.__name__} does not take. "
            f"{remedy}"
        ) from e
