#!/usr/bin/env python3
"""Decode an ESPnet-format Whisper SOT checkpoint via ``openai-whisper``.

The released ``model.pth`` uses ESPnet state-dict naming
(``encoder.encoders.*``, ``decoder.decoders.*``). We remap those keys back to
openai-whisper's naming in memory and delegate inference to whisper's
``transcribe()`` pipeline (temperature fallback, compression-ratio /
log-prob thresholds, KV cache, ``ApplyTimestampRules``).

The SOT speaker-change separator emitted by the model is rewritten to
``<sc>`` in the output text so downstream tooling can split on it cleanly.

Usage::

    python local/decode.py exp/whisper-sot-small-ami \\
        --wav_scp data/test/wav.scp \\
        --out_subdir decode_test \\
        --whisper_model small --fp16
"""

import argparse
import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import torch
import whisper
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("decode")

LOG_EVERY = 100


# ---------------------------------------------------------------------------
# SOT-aware replacement for whisper.decoding.ApplyTimestampRules
# ---------------------------------------------------------------------------
# Whisper's stock ``ApplyTimestampRules.apply`` enforces global timestamp
# pairing and forbids any text token after a closing timestamp. That prevents
# the SOT/FIFO model from emitting the speaker-change separator between
# speaker blocks (its trained pattern is ``<|t_end|> <sep> <|t_start|>``).
# The patch below scopes the pairing rules to the *current speaker block*
# (tokens after the last separator) so the separator becomes a valid choice
# right after a closing timestamp.


def _install_sot_separator_patch(sep_id: int) -> None:
    import whisper.decoding as _wd

    if getattr(_wd.ApplyTimestampRules, "_sot_patched", False):
        return

    def _sot_apply(self, logits, tokens):
        sep_in_vocab = sep_id < logits.shape[-1]
        sep_orig = logits[:, sep_id].detach().clone() if sep_in_vocab else None

        if self.tokenizer.no_timestamps is not None:
            logits[:, self.tokenizer.no_timestamps] = float("-inf")
        ts0 = self.tokenizer.timestamp_begin
        eot = self.tokenizer.eot

        for k in range(tokens.shape[0]):
            seq = tokens[k, self.sample_begin :].tolist()
            last_sep_pos = None
            if sep_in_vocab:
                for i in range(len(seq) - 1, -1, -1):
                    if seq[i] == sep_id:
                        last_sep_pos = i
                        break
            if last_sep_pos is not None:
                current_block = seq[last_sep_pos + 1 :]
                just_after_sep = len(current_block) == 0
            else:
                current_block = seq
                just_after_sep = False

            # Right after a separator: force a timestamp; forbid consecutive
            # separators.
            if just_after_sep:
                logits[k, :ts0] = float("-inf")
                logits[k, eot] = 0.0
                if sep_in_vocab:
                    logits[k, sep_id] = float("-inf")
                continue

            # Pairing rules SCOPED to the current speaker block.
            last_ts = len(current_block) >= 1 and current_block[-1] >= ts0
            penul_ts = len(current_block) < 2 or current_block[-2] >= ts0
            if last_ts:
                if penul_ts:
                    # Two consecutive timestamps -> text must follow.
                    logits[k, ts0:] = float("-inf")
                    if sep_in_vocab:
                        logits[k, sep_id] = float("-inf")
                else:
                    # Single closing timestamp -> suppress text below eot,
                    # restore the separator so a speaker change is available
                    # alongside EOT / next opening timestamp.
                    logits[k, :eot] = float("-inf")
                    if sep_in_vocab:
                        logits[k, sep_id] = sep_orig[k]

            # Non-decreasing timestamps within the current block.
            blk = [t for t in current_block if t >= ts0]
            if blk:
                ts_last = blk[-1] if (last_ts and not penul_ts) else blk[-1] + 1
                logits[k, ts0:ts_last] = float("-inf")

        # Forced initial timestamp (mirrors stock rule).
        if tokens.shape[1] == self.sample_begin:
            logits[:, :ts0] = float("-inf")
            if self.max_initial_timestamp_index is not None:
                last_allowed = ts0 + self.max_initial_timestamp_index
                logits[:, last_allowed + 1 :] = float("-inf")

        # Adaptive sampling: if logsumexp of timestamp logits exceeds the max
        # text logit, force a timestamp. Biases the model to *continue* within
        # a speaker block when its timestamp confidence is high; without it,
        # the model emits the separator too eagerly after the first closing
        # timestamp and DER suffers.
        import torch.nn.functional as _F

        logprobs = _F.log_softmax(logits.float(), dim=-1)
        for k in range(tokens.shape[0]):
            if logprobs[k, ts0:].logsumexp(dim=-1) > logprobs[k, :ts0].max():
                logits[k, :ts0] = float("-inf")

    _wd.ApplyTimestampRules.apply = _sot_apply
    _wd.ApplyTimestampRules._sot_patched = True


# ---------------------------------------------------------------------------
# State-dict remap: ESPnet -> openai-whisper
# ---------------------------------------------------------------------------


def remap_espnet_to_whisper(
    espnet_state: Dict[str, torch.Tensor],
) -> "OrderedDict[str, torch.Tensor]":
    """Strip ESPnet wrapper prefixes (``encoder.encoders.``, ``decoder.decoders.``)
    and merge ``ExpandedTokenEmbedding`` back into a single flat tensor that
    ``whisper.model.TextDecoder`` expects. The SOT separator was trained as a
    plain BPE pair already in the base vocabulary, so the optional ``add_emb``
    row is unused and only ``ori_emb`` is kept."""
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k, v in espnet_state.items():
        if "token_embedding.ori_emb" in k or "token_embedding.add_emb" in k:
            continue
        new_k = k
        if new_k.startswith("encoder.encoders."):
            new_k = "encoder." + new_k[len("encoder.encoders.") :]
        elif new_k.startswith("decoder.decoders."):
            new_k = "decoder." + new_k[len("decoder.decoders.") :]
        out[new_k] = v
    ori_key = "decoder.decoders.token_embedding.ori_emb.weight"
    if ori_key in espnet_state:
        out["decoder.token_embedding.weight"] = espnet_state[ori_key]
    return out


def load_whisper_model_from_espnet(
    espnet_pth: Path, whisper_model_size: str, device: str
) -> whisper.model.Whisper:
    """Initialize the openai-whisper architecture matching ``whisper_model_size``,
    then overwrite its weights with the converted ESPnet checkpoint."""
    logger.info(f"Initializing whisper.{whisper_model_size} architecture...")
    model = whisper.load_model(whisper_model_size, device="cpu")
    # Do NOT call model.half() here even with fp16=True; whisper's transcribe()
    # casts inputs/cache internally and pre-halving conflicts with that.
    logger.info(f"Loading ESPnet weights from {espnet_pth}")
    espnet_sd = torch.load(espnet_pth, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(
        remap_espnet_to_whisper(espnet_sd), strict=False
    )
    if missing:
        logger.warning(f"Missing {len(missing)} keys (first 3: {missing[:3]})")
    if unexpected:
        logger.warning(f"Unexpected {len(unexpected)} keys (first 3: {unexpected[:3]})")
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def read_speaker_change_symbol(model_dir: Path) -> str:
    """Read the speaker-change separator string from the checkpoint bundle's
    ``config.yaml`` (``preprocessor_conf.speaker_change_symbol``)."""
    cfg = yaml.safe_load(open(model_dir / "config.yaml"))
    sym = cfg.get("preprocessor_conf", {}).get("speaker_change_symbol")
    if sym is None:
        raise SystemExit(
            f"{model_dir / 'config.yaml'} lacks "
            "preprocessor_conf.speaker_change_symbol; pass --speaker_change_symbol"
        )
    return sym[0] if isinstance(sym, (list, tuple)) else sym


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("model_dir", type=Path, help="Directory with model.pth")
    p.add_argument(
        "--whisper_model",
        required=True,
        choices=whisper.available_models(),
        help="Architecture name for whisper.load_model",
    )
    p.add_argument("--wav_scp", type=Path, default=Path("data/test/wav.scp"))
    p.add_argument("--out_subdir", default="decode_inference")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--fp16", action="store_true", help="Run inference in fp16 (Whisper default)."
    )
    p.add_argument("--language", default="en")
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument(
        "--speaker_change_symbol",
        default=None,
        help="Speaker-change separator string. Defaults to the value in the "
        "checkpoint bundle's config.yaml (preprocessor_conf.speaker_change_symbol).",
    )
    args = p.parse_args()

    out_dir = args.model_dir / args.out_subdir / "1best_recog"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the speaker-change separator (from CLI or the bundle's config),
    # then derive its single-token id from the BPE vocabulary.
    sep_str = args.speaker_change_symbol or read_speaker_change_symbol(args.model_dir)
    sot_tokenizer = whisper.tokenizer.get_tokenizer(
        multilingual=True,
        task="transcribe",
        language=args.language,
    )
    sep_ids = sot_tokenizer.encode(sep_str)
    if len(sep_ids) != 1:
        raise SystemExit(
            f"Separator {sep_str!r} is not a single BPE token (got {sep_ids})"
        )
    sep_id = sep_ids[0]

    _install_sot_separator_patch(sep_id)
    logger.info(
        f"Patched whisper.decoding.ApplyTimestampRules for SOT separator "
        f"{sep_str!r} (id={sep_id})"
    )

    t0 = time.time()
    model = load_whisper_model_from_espnet(
        espnet_pth=args.model_dir / "model.pth",
        whisper_model_size=args.whisper_model,
        device=args.device,
    )
    logger.info(f"Model ready in {time.time() - t0:.1f}s")

    utts = []
    with open(args.wav_scp) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utts.append((parts[0], parts[1]))
    logger.info(f"Total utterances: {len(utts)}")

    def _build_text_sot(result: dict) -> str:
        parts = []
        for seg in result.get("segments", []):
            toks = list(seg.get("tokens", []))
            if toks:
                parts.append(sot_tokenizer.decode_with_timestamps(toks))
        return "".join(parts).replace(sep_str, " <sc> ")

    t_start = time.time()
    fail_count = 0
    with (
        open(out_dir / "text", "w") as f_text,
        open(out_dir / "text_sot", "w") as f_text_sot,
    ):
        for i, (uid, path) in enumerate(utts):
            try:
                audio = whisper.load_audio(path)  # 16kHz float32 mono
                result = model.transcribe(
                    audio,
                    language=args.language,
                    task="transcribe",
                    beam_size=args.beam_size,
                    fp16=args.fp16,
                    word_timestamps=False,
                    condition_on_previous_text=False,
                    # Allow the first emitted timestamp to be anywhere in the
                    # segment. The default 1.0s would force <|0.00|> even when
                    # speech starts >1s into the segment, inflating
                    # false-alarm DER on short/silent-prefix utterances.
                    max_initial_timestamp=None,
                )
                text = " ".join(
                    result["text"].strip().replace(sep_str, " <sc> ").split()
                )
                f_text.write(f"{uid} {text}\n")
                f_text_sot.write(f"{uid} {_build_text_sot(result)}\n")
            except Exception as e:
                fail_count += 1
                logger.warning(f"[{uid}] FAILED: {type(e).__name__}: {e}")
                f_text.write(f"{uid} \n")
                f_text_sot.write(f"{uid} \n")
            if (i + 1) % LOG_EVERY == 0:
                f_text.flush()
                f_text_sot.flush()
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                logger.info(
                    f"[{i + 1}/{len(utts)}] {rate:.2f} utt/s "
                    f"ETA {(len(utts) - i - 1) / rate / 60:.1f} min"
                )

    total = time.time() - t_start
    logger.info(
        f"Done. {len(utts)} utts in {total / 60:.1f} min "
        f"({len(utts) / total:.2f} utt/s). failures: {fail_count}"
    )


if __name__ == "__main__":
    sys.exit(main())
