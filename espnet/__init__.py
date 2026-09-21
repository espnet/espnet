"""ESPnet's top-level entry points.

``pip install espnet`` installs ``espnet2``, ``espnet3`` and ``egs3``, so
until now ``import espnet`` - the obvious first line for anyone who read the
distribution's name - raised ``ModuleNotFoundError``. This package is that
line: a version string and one loader.

    >>> import espnet
    >>> asr = espnet.load("espnet/owsm_ctc_v4_1B")           # doctest: +SKIP
    >>> tts = espnet.load("espnet/kan-bayashi_ljspeech_vits")  # doctest: +SKIP

Nothing heavy is imported here. ``torch`` and ``espnet2`` are pulled in by
:func:`load`, not by ``import espnet``, so a program that only wants
``espnet.__version__`` pays nothing for it.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, NoReturn, Optional, Tuple

__all__ = ["TASKS", "load", "__version__"]

try:
    __version__ = version("espnet")
except PackageNotFoundError:  # a source tree that was never pip-installed
    __version__ = "unknown (espnet is not installed; running from a source tree)"


# The espnet2 inference class that serves each task, by the espnet2 task name
# the recipes and the model tags already use. Values are (module, class) so
# that naming a task costs no import.
TASKS: Dict[str, Tuple[str, str]] = {
    "asr": ("espnet2.bin.asr_inference", "Speech2Text"),
    # one class for both kinds of OWSM checkpoint: it reads the training
    # config and builds either the encoder-decoder or the CTC-only model.
    # `__call__` searches, `best_path()` decodes on the CTC head alone, and
    # `decode_long()` handles a recording of any length.
    "s2t": ("espnet2.bin.s2t_inference", "Speech2Text"),
    "tts": ("espnet2.bin.tts_inference", "Text2Speech"),
    "enh": ("espnet2.bin.enh_inference", "SeparateSpeech"),
    "spk": ("espnet2.bin.spk_inference", "Speech2Embedding"),
    "diar": ("espnet2.bin.diar_inference", "DiarizeSpeech"),
}

# Hugging Face labels that name an espnet task, most specific first: the first
# one a repository carries decides. Order is what resolves the overlaps - an
# espnet speaker model is published as `audio-classification` with a
# `speaker-*` tag beside it, and `audio-classification` on its own says only
# "some classifier", so it is deliberately absent here rather than mapped to
# a guess. Each label is matched against the repository's `pipeline_tag` and
# its tags together, because older repositories set only the latter.
_HUB_LABELS: Tuple[Tuple[str, str], ...] = (
    ("speaker-verification", "spk"),
    ("speaker-recognition", "spk"),
    ("voice-activity-detection", "diar"),
    ("text-to-speech", "tts"),
    ("text-to-audio", "tts"),
    ("audio-to-audio", "enh"),
    # ASR and S2T share every Hub label there is; _asr_or_s2t() splits them.
    ("automatic-speech-recognition", ""),
    ("automatic-speech-translation", ""),
)


def _task_names() -> str:
    return ", ".join(sorted(TASKS))


def _hub_labels(model_tag: str) -> set:
    """Every Hugging Face label on a repository, or an empty set."""
    from huggingface_hub import model_info
    from huggingface_hub.utils import HfHubHTTPError

    try:
        info = model_info(model_tag)
    except (HfHubHTTPError, OSError, ValueError):
        # a local directory, a plain URL, a private or missing repository:
        # all of them mean "no metadata to read", and the caller says so
        return set()
    return {label for label in [info.pipeline_tag, *(info.tags or [])] if label}


def _asr_or_s2t(model_tag: str) -> Optional[str]:
    """Tell an OWSM-style S2T model from a classic ASR model.

    The Hub calls both `automatic-speech-recognition`, but the two inference
    classes take differently named constructor arguments, and a packed model's
    ``meta.yaml`` names them: ``s2t_train_config`` for one, ``asr_train_config``
    for the other. Reading that one small file is cheaper than downloading the
    checkpoint to find out.
    """
    import yaml
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

    try:
        path = hf_hub_download(model_tag, "meta.yaml")
        with open(path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)
    except (HfHubHTTPError, OSError, ValueError, yaml.YAMLError):
        return None
    if not isinstance(meta, dict):
        return None
    keys = {*meta.get("files", {}), *meta.get("yaml_files", {})}
    return "s2t" if any(key.startswith("s2t_") for key in keys) else "asr"


def _infer_task(model_tag: str) -> str:
    """Work out which espnet task a published model serves.

    Decided by the model's own Hugging Face metadata: its ``pipeline_tag`` and
    its tags are matched against :data:`_HUB_LABELS`, most specific label
    first. Speech recognition and speech-to-text translation carry the same
    Hub labels, so those two are split by the ``meta.yaml`` inside the
    repository - see :func:`_asr_or_s2t`.
    """
    labels = _hub_labels(model_tag)
    for label, task in _HUB_LABELS:
        if label in labels:
            return task or _asr_or_s2t(model_tag) or _unknown_task(model_tag)
    return _unknown_task(model_tag)


def _unknown_task(model_tag: str) -> NoReturn:
    raise ValueError(
        f"cannot tell what task {model_tag} is for from its Hugging Face "
        f"metadata. Pass it yourself, e.g. "
        f"espnet.load({model_tag!r}, task='asr'); the tasks are {_task_names()}."
    )


def load(
    model_tag: str,
    task: Optional[str] = None,
    device: str = "cpu",
    **kwargs: Any,
):
    """Load a published ESPnet model and return its inference object.

    The model is downloaded from https://huggingface.co/espnet on first use
    and kept in the ``espnet_model_zoo`` cache, so a second call is local.

    Args:
        model_tag: A tag from the espnet organisation, e.g.
            ``"espnet/owsm_ctc_v4_1B"``.
        task: One of ``asr``, ``s2t``, ``tts``, ``enh``, ``spk``, ``diar``.
            When omitted it is inferred from the model's own Hugging Face
            metadata: the repository's ``pipeline_tag`` and tags name the
            task, and where those cannot separate speech recognition from
            speech-to-text translation - the Hub labels both
            ``automatic-speech-recognition`` - the ``meta.yaml`` in the
            repository does, because the two classes take differently named
            constructor arguments. An explicit ``task`` always wins and skips
            the lookup entirely.
        device: ``cpu``, ``mps``, ``cuda`` or ``cuda:<n>``.
        **kwargs: Passed on to the inference class, e.g. ``beam_size=1``.

    Returns:
        The espnet2 inference object for the task: ``Speech2Text`` (``asr``
        and ``s2t`` have one of their own), ``Text2Speech``,
        ``SeparateSpeech``, ``Speech2Embedding`` or ``DiarizeSpeech``.

        Calling an ``s2t`` object runs a search, which on a CTC-only
        checkpoint such as OWSM-CTC is a CTC prefix beam search and is slow;
        its ``best_path()`` is the argmax decoding that was
        ``Speech2TextGreedySearch``, and is what an interactive first look
        wants.

    Raises:
        ValueError: ``task`` is not an espnet task, or none was given and the
            model's metadata does not name one.
        ModelTagError: the tag names a model this task cannot load.
    """
    if task is None:
        task = _infer_task(model_tag)
    elif task not in TASKS:
        raise ValueError(
            f"unknown task {task!r}: espnet.load takes one of {_task_names()}."
        )

    import importlib

    from espnet2.utils.pretrained import build_pretrained

    module_name, class_name = TASKS[task]
    loader = getattr(importlib.import_module(module_name), class_name)
    return build_pretrained(
        loader,
        model_tag,
        device,
        f"espnet.load(task={task!r})",
        f"Pass a {task} model, or name the task the tag really serves: "
        f"espnet.load({model_tag!r}, task=...) takes {_task_names()}.",
        **kwargs,
    )
