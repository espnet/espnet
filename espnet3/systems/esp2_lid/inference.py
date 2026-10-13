"""Language identification inference.

:class:`Speech2Language` runs a trained LID model on waveforms.
:class:`Inference` exposes it through :mod:`espnet3.api.inference`: the
recipe's ``infer`` stage runs it, and a packed LID model can be loaded with
``espnet3.api.inference.load`` and called like any other system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from espnet3.api.inference import Audio, Field
from espnet3.systems.base.backend_inference import BackendInference, parse_rate
from espnet3.systems.esp2_lid.task import LIDTask

# The rate of ESPnet2's default and S3PRL frontends when the training config
# names none.
DEFAULT_FRONTEND_RATE = 16000


class Speech2Language:
    """Predict language codes and optionally return normalized embeddings.

    Args:
        lid_train_config: ESPnet model configuration path.
        lid_model_file: Trained model or ESPnet3 checkpoint path.
        lang2utt: Language mapping in the model's class order.
        device: Torch device used for inference.
        dtype: Floating-point inference dtype.
        extract_embd: Return dictionaries with ``hyp`` and ``embedding`` instead
            of language strings. Embeddings are L2-normalized, as in ESPnet2.
            A waveform returns one dictionary; a waveform list returns a list.
    """

    def __init__(
        self,
        lid_train_config: str | Path,
        lid_model_file: str | Path,
        lang2utt: str | Path,
        device: str = "cpu",
        dtype: str = "float32",
        extract_embd: bool = False,
    ) -> None:
        """Load the trained model and its language mapping."""
        self.device = torch.device(device)
        self.extract_embd = extract_embd
        self.dtype = getattr(torch, dtype)

        self.model, self.lid_train_args = LIDTask.build_model_from_file(
            config_file=lid_train_config,
            model_file=lid_model_file,
            device=device,
        )
        self.model.to(dtype=self.dtype)
        self.model.eval()
        self.languages = self._load_languages(lang2utt)

    @staticmethod
    def _load_languages(lang2utt: str | Path) -> list[str]:
        """Read language codes in the checkpoint class order."""
        return [
            line.split(maxsplit=1)[0]
            for line in Path(lang2utt).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _prepare_speech(self, speech) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Pad mono waveforms and retain their lengths for batched inference."""
        is_batch = isinstance(speech, (list, tuple))
        items = list(speech) if is_batch else [speech]
        if not items:
            raise ValueError("speech batch must not be empty")

        waveforms = []
        for item in items:
            waveform = torch.as_tensor(np.asarray(item), dtype=self.dtype)
            if waveform.ndim != 1:
                raise ValueError(
                    "Each speech input must be a one-dimensional waveform, "
                    f"got shape={tuple(waveform.shape)}"
                )
            if waveform.numel() == 0:
                raise ValueError("speech input must not be empty")
            waveforms.append(waveform)

        lengths = torch.tensor(
            [waveform.numel() for waveform in waveforms],
            dtype=torch.long,
            device=self.device,
        )
        padded = pad_sequence(waveforms, batch_first=True).to(self.device)
        return padded, lengths, is_batch

    @torch.inference_mode()
    def __call__(self, speech):
        """Return language strings, or hyp/embedding dictionaries when enabled.

        Args:
            speech: One one-dimensional waveform, or a list of waveforms.

        Returns:
            One language string when ``extract_embd=False`` (default), or a
            dictionary containing ``hyp`` and a CPU NumPy ``embedding`` when
            true. Batched input returns a list of the corresponding values.

        Raises:
            ValueError: If a waveform is empty or not one-dimensional.

        Example:
            >>> predictor = Speech2Language("exp/config.yaml", "exp/model.pth",
            ...                             "exp/stats/train/lang2utt")
            >>> language = predictor(speech)
            >>> languages = predictor([speech, other_speech])
        """
        padded, lengths, is_batch = self._prepare_speech(speech)
        embeddings, predictions = self.model(
            speech=padded,
            speech_lengths=lengths,
            lid_labels=None,
            extract_embd=True,
        )
        indices = predictions.detach().cpu().reshape(-1).tolist()
        languages = [self.languages[index] for index in indices]
        if self.extract_embd:
            embeddings = torch.nn.functional.normalize(embeddings.float(), p=2, dim=1)
            results = [
                {"hyp": language, "embedding": embedding.detach().cpu().numpy()}
                for language, embedding in zip(languages, embeddings)
            ]
            return results if is_batch else results[0]
        return languages if is_batch else languages[0]


class Inference(BackendInference):
    """Identify the language of speech through :mod:`espnet3.api.inference`.

    Wraps :class:`Speech2Language`. Built three ways, all ending in
    ``self.backend`` (see :class:`BackendInference`):

    - ``Inference.from_pretrained(tag_or_dir, device=...)`` from a
      ``pack_model`` bundle or Hub tag, including bundles whose
      ``inference.yaml`` names ``Speech2Language`` as the model;
    - ``Inference(lid_train_config=..., lid_model_file=..., lang2utt=...)``
      from ``Speech2Language``'s own arguments;
    - ``Inference(speech2language)`` around one already built.

    Called with one utterance (a path, a ``(rate, samples)`` pair, an array or
    an :class:`~espnet3.api.inference.Audio`) it returns ``{"language": str}``.

    Examples:
        >>> from espnet3.api.inference import load
        >>> model = load("exp/training/model_pack", system="esp2_lid")
        >>> model("utt.wav")
        {'language': 'jpn'}
        >>> model.batch([{"speech": first}, {"speech": second}])
        [{'language': 'jpn'}, {'language': 'eng'}]
    """

    backend_class = "espnet3.systems.esp2_lid.inference.Speech2Language"
    inputs = (Field("speech", "audio", "Speech"),)
    outputs = (Field("language", "text", "Language"),)

    @property
    def sample_rate(self) -> int:
        """Return the frontend rate: ``frontend_conf.fs``, else 16 kHz.

        A backend that is not a :class:`Speech2Language` (no
        ``lid_train_args``) falls back to the base class, which reads a
        ``sample_rate`` or ``fs`` attribute or raises.
        """
        args = getattr(self.backend, "lid_train_args", None)
        if args is None:
            return super().sample_rate
        conf = getattr(args, "frontend_conf", None) or {}
        return parse_rate(conf.get("fs", DEFAULT_FRONTEND_RATE))

    def run(self, speech: Audio) -> Mapping[str, Any]:
        """Return the predicted language of one utterance.

        Args:
            speech: The utterance, at :attr:`sample_rate`, one channel.

        Returns:
            ``{"language": str}``, an ISO 639-3 code in the training inventory.
        """
        return {"language": _language(self.backend(speech.array))}

    def run_batch(self, items: Sequence[Mapping[str, Any]]) -> Sequence[Mapping]:
        """Identify several utterances in one padded forward pass.

        Args:
            items: Mappings with a ``speech`` :class:`Audio` each.

        Returns:
            One ``{"language": str}`` per item, in order.
        """
        predictions = self.backend([item["speech"].array for item in items])
        return [{"language": _language(prediction)} for prediction in predictions]


def _language(prediction: Any) -> str:
    """Return the language from a string or an ``extract_embd`` dictionary."""
    return prediction["hyp"] if isinstance(prediction, dict) else prediction
