"""Bagpiper's native single-dialogue decoder for the ESPnet3 inference runner."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from espnet2.bin.speechlm_inference import (
    DEFAULT_TAG,
    LocalBagpiper,
    local_decoding_options,
)
from espnet3.contract.kinds.audio import Audio


class BagpiperInference:
    """Load a Bagpiper checkpoint for audio understanding or generation.

    Args:
        mode: ``understand`` answers a prompt about audio. ``generate``
            renders the audio described by a prompt.
        tag_or_dir: Hugging Face model ID or local release directory.
        train_config: Path to the native model configuration when supplied
            separately from the release.
        checkpoint: Path to the exported native weight file.
        device: Device for inference, as accepted by LocalBagpiper.
        dtype: Override the release's inference dtype.
        attn_implementation: Override the release's attention backend.
        allow_base: Allow inference from the pretraining checkpoint.
        decoding: Overrides for the selected checkpoint's decoding YAML.
            Use ``max_steps`` for the maximum length of each decoded segment.
    """

    def __init__(
        self,
        mode: str,
        tag_or_dir: str | Path = DEFAULT_TAG,
        train_config: str | Path | None = None,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        dtype: str | None = None,
        attn_implementation: str | None = None,
        allow_base: bool = False,
        decoding: Mapping | None = None,
    ) -> None:
        """Load the native decoder with the requested generation settings."""
        if mode not in ("understand", "generate"):
            raise ValueError("Bagpiper mode must be 'understand' or 'generate'")
        self.generate_audio = mode == "generate"
        self.decoding = local_decoding_options(decoding)
        self.model = LocalBagpiper.from_pretrained(
            tag_or_dir,
            device=device,
            dtype=dtype,
            train_config=train_config,
            checkpoint=checkpoint,
            allow_base=allow_base,
            attn_implementation=attn_implementation,
        )

    def __call__(
        self, prompt: str, speech: Audio | None = None, system: str | None = None
    ) -> dict[str, str | Audio]:
        """Run one request with the checkpoint's trained dialogue ordering.

        Args:
            prompt: User question or description of the sound to generate.
            speech: Input recording for understanding, at its source rate.
            system: The recorded system turn. Leave it out for dialogues
                that start with a user turn.

        Returns:
            Assistant text on one line for the runner's text SCP. Generation
            also returns the waveform at the codec's own sample rate.

        Raises:
            ValueError: The request's input audio disagrees with the mode.
            RuntimeError: Generation completed without the declared audio output.
        """
        if not isinstance(prompt, str):
            raise TypeError("Pass one prompt at a time (set batch_size to null)")
        if self.generate_audio and speech is not None:
            raise ValueError("Bagpiper generation expects a text prompt without speech")
        if not self.generate_audio and speech is None:
            raise ValueError("Bagpiper understanding requires input speech")
        dialogue = []
        if system is not None:
            dialogue.append(("system", "text", system))
        if speech is not None:
            # Audio keeps mono as a vector. The native preprocessor expects
            # (channels, samples), including a channel axis for mono.
            waveform = speech.array
            if waveform.ndim == 1:
                waveform = waveform[None, :]
            dialogue.append(("user", "audio", (waveform, speech.rate)))
        dialogue.append(("user", "text", prompt))

        text, audio = self.model.infer_dialogue(
            dialogue, audio_out=self.generate_audio, decoding=self.decoding
        )
        result: dict[str, str | Audio] = {"text": " ".join(text.splitlines())}
        if self.generate_audio:
            if audio is None or not audio[0].size:
                raise RuntimeError(
                    "Bagpiper completed generation without emitting audio"
                )
            result["audio"] = Audio(*audio)
        return result
