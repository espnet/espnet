"""BEATs tokenization model used by the SSL ``infer`` stage."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from espnet2.beats.tokenizer import BeatsRandomTokenizer, BeatsTokenizer
from espnet2.beats.utils import DEFAULT_FBANK_MEAN, DEFAULT_FBANK_STD
from espnet2.legacy.nets.pytorch_backend.nets_utils import pad_list

logger = logging.getLogger(__name__)


class BeatsTokenizationModel:
    """Callable that converts audio or fbank features into BEATs token ids.

    The SSL ``infer`` stage uses this model to produce the discrete targets
    consumed by BEATs encoder training. Two tokenizers are supported:

    - ``tokenizer_ckpt_path`` is ``None``: a random-projection tokenizer
      (``BeatsRandomTokenizer``). This is iteration 0 of BEATs pre-training.
      Its weights are fully determined by ``tokenizer_config.seed``.
    - ``tokenizer_ckpt_path`` points to a portable tokenizer checkpoint
      exported by ``train_tokenizer`` (``BeatsTokenizer``). This is used for
      iterations greater than 0.

    Input contract (controlled by ``waveform_input``):

    - ``waveform_input: true``: each item is a mono waveform of shape
      ``(num_samples,)`` with float values in ``[-1, 1]``, sampled at 16 kHz.
      Int16-scaled input (values outside ``[-1, 1]``) is rescaled with a
      warning, matching ``espnet2.beats.audio_tokenizer.AudioTokenizer``.
    - ``waveform_input: false``: each item is a Kaldi-style 128-bin log-mel
      fbank of shape ``(num_frames, 128)``, not normalized.

    Args:
        tokenizer_ckpt_path: Portable BEATs tokenizer checkpoint
            (``{"model": ..., "cfg": ...}``). ``None`` selects the random
            tokenizer.
        tokenizer_config: Optional tokenizer config overrides. For the random
            tokenizer this should contain ``seed`` (egs2 uses ``45``) and may
            override ``quant_n``/``quant_dim``.
        fbank_mean: Global fbank mean used for input normalization.
        fbank_std: Global fbank standard deviation used for input
            normalization.
        waveform_input: Whether inputs are waveforms (``True``) or fbank
            features (``False``).
        device: Torch device string, injected by the inference provider.

    Examples:
        Configure iteration-0 tokenization in ``inference.yaml``:

        .. code-block:: yaml

            model:
              _target_: espnet3.systems.ssl.tokenization_model.BeatsTokenizationModel
              tokenizer_ckpt_path: null
              tokenizer_config:
                seed: 45
              fbank_mean: 15.66439
              fbank_std: 6.38312
              waveform_input: false

        Call it directly:

        >>> model = BeatsTokenizationModel(tokenizer_config={"seed": 45})
        >>> codes = model(np.zeros(16000, dtype=np.float32))  # doctest: +SKIP
        >>> codes.shape  # one code per 16x16 fbank patch  # doctest: +SKIP
        (48,)
    """

    def __init__(
        self,
        tokenizer_ckpt_path: str | None = None,
        tokenizer_config: Dict[str, Any] | None = None,
        fbank_mean: float = DEFAULT_FBANK_MEAN,
        fbank_std: float = DEFAULT_FBANK_STD,
        waveform_input: bool = False,
        device: str = "cpu",
    ) -> None:
        """Build the random or trained BEATs tokenizer on ``device``."""
        tokenizer_config = dict(tokenizer_config or {})
        if tokenizer_ckpt_path is None:
            tokenizer = BeatsRandomTokenizer(
                tokenizer_config=tokenizer_config,
                fbank_mean=fbank_mean,
                fbank_std=fbank_std,
            )
            self.codebook_size = tokenizer.config.quant_n
        else:
            tokenizer = BeatsTokenizer(
                beats_tokenizer_ckpt_path=str(tokenizer_ckpt_path),
                tokenizer_config=tokenizer_config or None,
                fbank_mean=fbank_mean,
                fbank_std=fbank_std,
            )
            self.codebook_size = tokenizer.quantize.num_tokens
        self.device = torch.device(device)
        self.tokenizer = tokenizer.to(self.device).eval()
        self.waveform_input = waveform_input

    def _to_tensor(self, item) -> torch.Tensor:
        tensor = torch.as_tensor(np.asarray(item), dtype=torch.float32)
        if self.waveform_input:
            if tensor.dim() != 1:
                raise ValueError(
                    "waveform_input=True expects a mono waveform of shape "
                    f"(num_samples,), got {tuple(tensor.shape)}."
                )
            if tensor.numel() > 0 and tensor.abs().max() > 1.0:
                logger.warning("Waveform is not in [-1, 1]; rescaling from int16.")
                tensor = tensor / 2**15
        elif tensor.dim() != 2:
            raise ValueError(
                "waveform_input=False expects fbank features of shape "
                f"(num_frames, num_mel_bins), got {tuple(tensor.shape)}."
            )
        return tensor

    @torch.no_grad()
    def __call__(self, speech) -> np.ndarray | List[np.ndarray]:
        """Tokenize one item or a list of items.

        Args:
            speech: One waveform/fbank array, or a list of them for batched
                inference. Items in a batch may have different lengths; they
                are zero-padded and the padding is masked out.

        Returns:
            np.ndarray | List[np.ndarray]: ``int64`` code ids of shape
            ``(num_patches,)`` for a single item, or one such array per item
            when ``speech`` is a list.
        """
        batched = isinstance(speech, (list, tuple))
        items: Sequence[Any] = speech if batched else [speech]
        tensors = [self._to_tensor(item) for item in items]
        lengths = torch.tensor([t.shape[0] for t in tensors], dtype=torch.long)
        xs_pad = pad_list(tensors, 0.0).to(self.device)
        encoded = self.tokenizer.encode(
            xs_pad, lengths.to(self.device), waveform_input=self.waveform_input
        )
        codes = encoded["codes"].cpu().numpy().astype(np.int64)
        code_lengths = encoded["code_lengths"].cpu().numpy()
        outputs = [code[: int(length)] for code, length in zip(codes, code_lengths)]
        return outputs if batched else outputs[0]


def build_target_output(data, model_output, idx):
    """Format tokenization results as ``target.scp`` records.

    Referenced from the SSL ``inference.yaml`` as ``output_fn``. Each record is
    keyed by the dataset item index (``idx_key: idx``) so the training dataset
    can join targets back to its samples without an ``utt_id`` field.

    Args:
        data: Dataset sample, or a list of samples for batched inference.
            Unused; accepted for the ``output_fn`` protocol.
        model_output: Code ids returned by :class:`BeatsTokenizationModel`,
            one array per item when batched.
        idx: Dataset index, or a list of indices when batched.

    Returns:
        dict | list[dict]: ``{"idx": <int>, "target": "<id> <id> ..."}`` for
        one item, or a list of such dicts for a batch.

    Examples:
        >>> build_target_output({}, np.array([3, 1, 2]), 7)
        {'idx': 7, 'target': '3 1 2'}
    """
    if isinstance(idx, (list, tuple)):
        return [
            build_target_output(item, output, index)
            for item, output, index in zip(data, model_output, idx)
        ]
    codes = np.asarray(model_output).reshape(-1)
    return {"idx": int(idx), "target": " ".join(str(int(c)) for c in codes)}
