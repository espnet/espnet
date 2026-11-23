"""Inference wrapper for the offline Sortformer diarization model.

Instantiated by the ESPnet3 inference stage (Hydra ``_target_``) and called as
``model(speech=<np.ndarray>)``; returns per-frame speaker probabilities
``(T, num_spk)`` as a numpy array. Builds the architecture from the saved
training config and loads weights from a plain ``.pth`` state-dict or a
PyTorch-Lightning ``.ckpt`` (``model.`` prefix is stripped automatically).

Selected in an inference config by Hydra ``_target_``::

    inference:
      _target_: espnet3.systems.diar.sortformer.inference_bin.SortformerDiarization
      train_config: exp/sortformer/config.yaml
      model_file: exp/sortformer/valid.acc.best.pth
      device: cuda
      threshold: 0.5

and driven by the ``infer`` stage::

    python run.py --stages infer \\
        --training_config conf/training.yaml \\
        --inference_config conf/inference.yaml
"""

from typing import Optional, Union

import numpy as np
import torch
import yaml

from espnet3.utils.task_utils import get_espnet_model

_TASK = "espnet3.systems.diar.task.SortformerDiarizationTask"


def _strip_prefix(state_dict, prefix="model."):
    """Strip ``prefix`` from state-dict keys when a Lightning ckpt is detected.

    Returns the dict unchanged if no key starts with ``prefix``.
    """
    if any(k.startswith(prefix) for k in state_dict):
        return {
            k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
        }
    return state_dict


class SortformerDiarization:
    """Callable Sortformer diarizer for ESPnet3 inference.

    Builds the Sortformer model from a saved training config (via the
    ``espnet3.systems.diar.task.SortformerDiarizationTask`` dotted path) and loads its
    weights, then exposes a simple ``model(speech)`` call returning per-frame
    speaker activity.

    Instantiated by the inference stage through Hydra ``_target_``::

        inference:
          _target_: espnet3.systems.diar.sortformer.inference_bin.SortformerDiarization
          train_config: exp/sortformer/config.yaml
          model_file: exp/sortformer/valid.acc.best.pth
          device: cuda
          threshold: 0.5

    Args:
        train_config: Path to the espnet3-saved (flattened) training YAML.
        model_file: Path to a ``.pth`` state-dict or a Lightning ``.ckpt``.
        device: Torch device to run on (e.g. ``"cpu"`` or ``"cuda"``).
        threshold: Optional probability threshold; if set, the returned
            activity is binarized, otherwise raw probabilities are returned.

    Raises:
        RuntimeError: If required weights are missing from ``model_file``
            (the feature-extraction buffers ``preprocessor.fb`` /
            ``preprocessor.window`` are allowed to be absent).
    """

    def __init__(
        self,
        train_config: str,
        model_file: str,
        device: str = "cpu",
        threshold: Optional[float] = None,
    ):
        with open(train_config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # The espnet3-saved config is flattened at the root; pass it straight to
        # the task builder (extra keys are ignored).
        self.model = get_espnet_model(_TASK, cfg)

        state = torch.load(model_file, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = _strip_prefix(state, "model.")
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        allowed = ("preprocessor.fb", "preprocessor.window")
        real_missing = [m for m in missing if not m.startswith(allowed)]
        if real_missing:
            raise RuntimeError(f"Missing keys when loading model: {real_missing}")

        self.device = device
        self.model.to(device).eval()
        self.threshold = threshold

    @torch.no_grad()
    def __call__(self, speech: Union[np.ndarray, list, torch.Tensor]) -> np.ndarray:
        """Diarize a single utterance.

        Args:
            speech: Mono waveform as a 1-D (or ``(1, T)``) array/list/tensor of
                samples.

        Returns:
            A ``(T, num_spk)`` numpy array of per-frame speaker probabilities,
            or 0/1 activity if ``threshold`` was set at construction time.

        Example:
            >>> diarizer = SortformerDiarization(train_config, model_file)
            >>> activity = diarizer(speech=wav)  # (T, num_spk)
        """
        # Accept either a torch.Tensor (possibly already on GPU) or array-like;
        # np.asarray() on a CUDA tensor would raise, so branch on the type.
        if isinstance(speech, torch.Tensor):
            wav = speech.to(dtype=torch.float32)
        else:
            wav = torch.as_tensor(np.asarray(speech), dtype=torch.float32)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.to(self.device)
        lengths = torch.tensor([wav.shape[1]], device=self.device)
        preds, plen = self.model.diarize(wav, lengths)
        preds = preds[0, : plen[0]].cpu().numpy()
        if self.threshold is not None:
            preds = (preds >= self.threshold).astype(np.float32)
        return preds
