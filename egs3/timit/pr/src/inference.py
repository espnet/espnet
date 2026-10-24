"""PhoneticXeus inference for the TIMIT phone recognition recipe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder
from espnet2.asr.frontend.cnn import CNNFrontend
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.legacy.nets.pytorch_backend.nets_utils import make_pad_mask

_FRONTEND_CONF = dict(
    norm_mode="layer_norm",
    conv_mode="standard",
    bias=True,
    normalize_audio=True,
    normalize_output=False,
    fs="16k",
)
_PREENCODER_OUTPUT_SIZE = 1024
_ENCODER_CONF = dict(
    output_size=1024,
    attention_heads=8,
    attention_layer_type="selfattn",
    pos_enc_layer_type="conv",
    rel_pos_type="latest",
    cgmlp_linear_units=4096,
    cgmlp_conv_kernel=31,
    use_linear_after_conv=False,
    gate_activation="identity",
    num_blocks=19,
    dropout_rate=0.1,
    positional_dropout_rate=0.1,
    attention_dropout_rate=0.1,
    input_layer=None,
    layer_drop_rate=0.0,
    linear_units=4096,
    positionwise_layer_type="linear",
    macaron_ffn=True,
    use_ffn=True,
    merge_conv_kernel=31,
)
# Encoder layers whose CTC posteriors are fed back into the encoder, matching
# the configuration the published weights were trained with.
_INTERCTC_LAYER_IDX = [4, 8, 12]


def _load_vocab(model_id: str) -> List[str]:
    """Fetch the IPA token list that belongs to a checkpoint.

    Taken from the same repository as the weights, so the vocabulary can never
    drift out of step with the CTC output layer it labels.

    Args:
        model_id: Hugging Face repository holding ``ipa_vocab.json``.

    Returns:
        Tokens ordered by id, so index ``i`` is the label of CTC output ``i``.
    """
    path = hf_hub_download(model_id, "ipa_vocab.json")
    token_to_id = json.loads(Path(path).read_text(encoding="utf-8"))
    tokens = [""] * len(token_to_id)
    for token, index in token_to_id.items():
        tokens[int(index)] = token
    return tokens


class PhoneticXeus(torch.nn.Module):
    """Run the published PhoneticXeus model over raw waveforms.

    Instantiated by the ``infer`` stage from ``conf/inference.yaml``, which
    passes ``device``. Call it with one utterance's waveform and it returns the
    greedy CTC transcript.

    Args:
        model_id: Hugging Face repository holding ``phoneticxeus_state_dict.pt``.
        device: Torch device string, injected by the inference provider.

    Raises:
        RuntimeError: If the checkpoint does not fit the rebuilt architecture,
            or if the encoder would not apply the intermediate CTC feedback.

    Examples:
        Selected from ``conf/inference.yaml`` with::

            model:
              _target_: src.inference.PhoneticXeus
              model_id: changelinglab/PhoneticXeus
    """

    def __init__(
        self,
        model_id: str = "changelinglab/PhoneticXeus",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.token_list = _load_vocab(model_id)
        self.blank_id = self.token_list.index("<blank>")

        self.frontend = CNNFrontend(**_FRONTEND_CONF)
        self.preencoder = LinearProjection(
            input_size=self.frontend.output_size(),
            output_size=_PREENCODER_OUTPUT_SIZE,
        )
        self.encoder = EBranchformerEncoder(
            input_size=_PREENCODER_OUTPUT_SIZE,
            interctc_layer_idx=list(_INTERCTC_LAYER_IDX),
            interctc_use_conditioning=True,
            **_ENCODER_CONF,
        )
        self.ctc = CTC(
            odim=len(self.token_list),
            encoder_output_size=_ENCODER_CONF["output_size"],
            dropout_rate=0.0,
            ctc_type="builtin",
            zero_infinity=True,
        )
        # Built here rather than by the encoder, matching how the published
        # model attaches it, so the checkpoint's weights have a home.
        self.encoder.conditioning_layer = torch.nn.Linear(
            len(self.token_list), _ENCODER_CONF["output_size"]
        )

        self._load_weights(model_id)

        self.device = torch.device(device)
        self.to(self.device).eval()

    def _load_weights(self, model_id: str) -> None:
        """Load the published weights and refuse a partial match.

        Args:
            model_id: Hugging Face repository id.

        Raises:
            RuntimeError: If any parameter is missing or unexpected. The
                upstream loader tolerates both, which is how a silently
                half-loaded model goes unnoticed.
        """
        checkpoint = hf_hub_download(model_id, "phoneticxeus_state_dict.pt")
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = {
                key.replace("net.", ""): value
                for key, value in state_dict["state_dict"].items()
                if key.startswith("net.")
            }
        info = self.load_state_dict(state_dict, strict=False)
        if info.missing_keys or info.unexpected_keys:
            raise RuntimeError(
                f"{model_id} does not match the rebuilt architecture. "
                f"Missing: {sorted(info.missing_keys)[:5]}. "
                f"Unexpected: {sorted(info.unexpected_keys)[:5]}."
            )

    def encode(self, speech: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Run the frontend, preencoder and encoder over one batch.

        Args:
            speech: Waveforms, shape ``(batch, samples)``.
            lengths: Sample counts per utterance, shape ``(batch,)``.

        Returns:
            Encoder output, shape ``(batch, frames, 1024)``.
        """
        feats, feats_lengths = self.frontend(speech, lengths)
        feats, feats_lengths = self.preencoder(feats, feats_lengths)
        # `ctc` is required for the feedback: the encoder calls `ctc.softmax`
        # on each intermediate output before adding it back into the stream.
        encoder_out, _, _ = self.encoder(
            feats,
            feats_lengths,
            masks=make_pad_mask(feats_lengths).to(feats.device),
            ctc=self.ctc,
        )
        if isinstance(encoder_out, tuple):
            # With interctc layers the encoder also returns the intermediates.
            encoder_out = encoder_out[0]
        return encoder_out

    @torch.no_grad()
    def forward(self, speech: np.ndarray | torch.Tensor) -> Dict[str, str]:
        """Transcribe one utterance into IPA phones.

        Args:
            speech: A 16 kHz mono waveform, shape ``(samples,)``.

        Returns:
            ``{"hyp": <concatenated IPA>, "tokens": <slash-separated phones>}``.
            ``hyp`` drops the special tokens and is what the metrics score;
            ``tokens`` keeps the model's own phone boundaries for inspection.
        """
        if not isinstance(speech, torch.Tensor):
            speech = torch.from_numpy(np.asarray(speech, dtype=np.float32))
        speech = speech.to(device=self.device, dtype=torch.float32)
        if speech.dim() == 1:
            speech = speech.unsqueeze(0)
        lengths = torch.full(
            (speech.size(0),), speech.size(1), dtype=torch.long, device=self.device
        )

        logits = self.ctc.ctc_lo(self.encode(speech, lengths))
        ids = logits.argmax(dim=-1)[0]
        # Standard CTC collapse: drop repeats, then drop blanks.
        keep = torch.ones_like(ids, dtype=torch.bool)
        keep[1:] = ids[1:] != ids[:-1]
        keep &= ids != self.blank_id
        tokens = [self.token_list[i] for i in ids[keep].tolist()]
        phones = [t for t in tokens if not (t.startswith("<") and t.endswith(">"))]
        return {"hyp": "".join(phones).strip(), "tokens": "/".join(tokens)}


def build_output(data: Any, model_output: Any, idx: Any) -> Any:
    """Shape one inference result into the fields written to SCP files.

    Args:
        data: The dataset sample, or a list of them when batched.
        model_output: The model's return value, matching ``data``.
        idx: The dataset index, or a list of them when batched.

    Returns:
        A mapping with ``utt_id``, ``hyp`` and ``ref``, or a list of them.

    Notes:
        ``ref`` comes from the dataset sample: the framework writes no
        reference file of its own. ``utt_id`` is the item index because recipe
        samples must not carry one; the manifest is sorted, so an index maps
        back to an utterance id.
    """
    if isinstance(data, list):
        return [build_output(d, o, i) for d, o, i in zip(data, model_output, idx)]
    return {
        "utt_id": str(idx),
        "hyp": model_output["hyp"],
        "ref": data["text"],
    }
