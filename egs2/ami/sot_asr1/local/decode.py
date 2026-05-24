#!/usr/bin/env python3
"""Decode an ESPnet-format Whisper SOT checkpoint via ``openai-whisper``.

The released ``model.pth`` is in ESPnet state-dict naming (encoder.encoders.*,
decoder.decoders.*) because it was loaded through ESPnet's Whisper modules.
The simplest way to get paper-matching cpWER is to delegate inference to
``openai-whisper``'s well-tested ``transcribe()`` pipeline (which has the full
temperature-fallback + compression-ratio + log-prob-threshold + KV cache +
``ApplyTimestampRules`` machinery). To do that we remap the ESPnet keys back
to openai-whisper's naming in memory, then ``model.load_state_dict``.

For multi-talker SOT output, the ``????`` separator is a regular BPE pair
(token id 25629); ``transcribe()`` emits it inline in the text. We rewrite it
to ``<sc>`` to match the reference text format used by ``evaluate_sot.py``.

Usage::

    python local/decode.py exp/converted_sst_small \\
        --wav_scp data/test/wav.scp \\
        --out_subdir decode_test_fallback \\
        --whisper_model small
"""

import argparse
import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import whisper

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger("decode")


# ---------------------------------------------------------------------------
# SOT-aware replacement for whisper's ApplyTimestampRules
# ---------------------------------------------------------------------------
# Whisper's stock ``ApplyTimestampRules.apply`` enforces *global* timestamp
# pairing and forbids any text after a closing timestamp. That makes it
# impossible for our SST/FIFO model to emit the ``????`` separator (BPE id
# 25629) between speaker blocks (its trained pattern is
# ``<|t_end|> ???? <|t_start|>``).
#
# This monkey-patch replaces ``ApplyTimestampRules.apply`` with a port of
# ``WhisperTimeStampLogitsProcessorCustom.__call__`` from the paper's
# TS-ASR-Whisper code (``src/models/dicow/utils.py``), simplified to the
# FIFO case (no spk_count / spk_rem / spk_id auxiliary tokens). Key change:
# rules are scoped to the *current speaker block* (tokens after the last
# separator) instead of the full sequence, and the separator itself becomes
# a valid choice right after a closing timestamp.

WHISPER_SOT_SEPARATOR_ID = 25629  # tiktoken-multilingual encoding of "????"


def _install_sot_separator_patch() -> None:
    """Replace whisper.decoding.ApplyTimestampRules.apply with an SOT-aware
    version that mirrors TS-ASR-Whisper's WhisperTimeStampLogitsProcessorCustom.
    """
    import whisper.decoding as _wd

    if getattr(_wd.ApplyTimestampRules, "_sot_patched", False):
        return  # idempotent

    SEP = WHISPER_SOT_SEPARATOR_ID

    def _sot_apply(self, logits, tokens):
        # Snapshot the original separator logits for later restoration.
        sep_in_vocab = SEP < logits.shape[-1]
        sep_orig = logits[:, SEP].detach().clone() if sep_in_vocab else None

        # Step 1: suppress <|notimestamps|> globally.
        if self.tokenizer.no_timestamps is not None:
            logits[:, self.tokenizer.no_timestamps] = float("-inf")

        ts0 = self.tokenizer.timestamp_begin
        eot = self.tokenizer.eot

        # Step 2: per-sample, block-scoped rules.
        for k in range(tokens.shape[0]):
            seq = tokens[k, self.sample_begin:].tolist()

            # Find last separator -> isolate the current speaker block.
            last_sep_pos = None
            if sep_in_vocab:
                for i in range(len(seq) - 1, -1, -1):
                    if seq[i] == SEP:
                        last_sep_pos = i
                        break
            if last_sep_pos is not None:
                current_block = seq[last_sep_pos + 1:]
                just_after_sep = len(current_block) == 0
            else:
                current_block = seq
                just_after_sep = False

            # 2a) Right after a separator -> force a timestamp; forbid another
            # consecutive separator.
            if just_after_sep:
                logits[k, :ts0] = float("-inf")
                logits[k, eot] = 0.0  # allow EOT as well
                if sep_in_vocab:
                    logits[k, SEP] = float("-inf")
                continue

            # 2b) Pairing rules SCOPED to the current speaker block.
            last_ts = len(current_block) >= 1 and current_block[-1] >= ts0
            penul_ts = len(current_block) < 2 or current_block[-2] >= ts0
            if last_ts:
                if penul_ts:
                    # Two consecutive timestamps -> text must follow.
                    logits[k, ts0:] = float("-inf")
                    if sep_in_vocab:
                        logits[k, SEP] = float("-inf")
                else:
                    # Single closing timestamp -> suppress text BELOW eot, but
                    # restore the separator score so a speaker change is
                    # available alongside EOT / next opening timestamp.
                    logits[k, :eot] = float("-inf")
                    if sep_in_vocab:
                        logits[k, SEP] = sep_orig[k]

            # 2c) Non-decreasing timestamps within the current block.
            blk = [t for t in current_block if t >= ts0]
            if blk:
                if last_ts and not penul_ts:
                    ts_last = blk[-1]
                else:
                    ts_last = blk[-1] + 1
                logits[k, ts0:ts_last] = float("-inf")

        # Step 3: forced initial timestamp (mirrors stock rule).
        if tokens.shape[1] == self.sample_begin:
            logits[:, :ts0] = float("-inf")
            if self.max_initial_timestamp_index is not None:
                last_allowed = ts0 + self.max_initial_timestamp_index
                logits[:, last_allowed + 1:] = float("-inf")

        # Step 4: adaptive sampling — if logsumexp(timestamp logits) exceeds
        # the max text logit, force a timestamp. Matches paper-side
        # WhisperTimeStampLogitsProcessorCustom step 4 and openai-whisper's
        # stock ApplyTimestampRules. This biases the model to *continue*
        # within a speaker block when it has high timestamp confidence
        # (multiple ``<|t|> text <|t|>`` pairs), and only switch speakers
        # when its text/separator confidence is genuinely higher. Without
        # this step, DER suffers because the model emits the separator too
        # eagerly after the first closing timestamp.
        import torch.nn.functional as _F
        logprobs = _F.log_softmax(logits.float(), dim=-1)
        for k in range(tokens.shape[0]):
            ts_logprob = logprobs[k, ts0:].logsumexp(dim=-1)
            text_max_logprob = logprobs[k, :ts0].max()
            if ts_logprob > text_max_logprob:
                logits[k, :ts0] = float("-inf")

    _wd.ApplyTimestampRules.apply = _sot_apply
    _wd.ApplyTimestampRules._sot_patched = True


# ---------------------------------------------------------------------------
# Reverse state-dict rename: ESPnet -> openai-whisper
# ---------------------------------------------------------------------------
# Conversion rules are the inverse of ``local/convert_hf_to_espnet.py``'s
# forward map. ESPnet's ``OpenAIWhisperEncoder/Decoder`` are essentially
# whisper.model.AudioEncoder/TextDecoder wrapped under ``.encoders/.decoders``,
# so we just strip those two prefixes; key names below the wrapper are
# already in openai-whisper's naming convention.
ESPNET_PREFIX_RULES = [
    ("encoder.encoders.", "encoder."),
    ("decoder.decoders.", "decoder."),
]


def remap_espnet_to_whisper(
    espnet_state: Dict[str, torch.Tensor]
) -> OrderedDict:
    """Strip ESPnet wrapper prefixes and merge the token embedding back into
    a single flat tensor that ``whisper.model.TextDecoder`` expects.

    ESPnet's ``ExpandedTokenEmbedding`` stores ``ori_emb`` (base vocab) and
    ``add_emb`` (added tokens like ``????``) separately. openai-whisper has
    a single ``decoder.token_embedding.weight`` of size ``(n_vocab, d_model)``.
    Our SOT models were trained with ``????`` as plain BPE (id 25629, in the
    base 51865 vocab), so the ``add_emb`` row is unused at inference — we
    keep only the original embedding and the final size matches whisper's
    ``n_vocab``.
    """
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    # Pass 1: copy + rename everything except the token embedding pair.
    for k, v in espnet_state.items():
        new_k = k
        for old, new in ESPNET_PREFIX_RULES:
            if new_k.startswith(old):
                new_k = new + new_k[len(old):]
                break
        # Skip the ExpandedTokenEmbedding parts (handled separately below).
        if "token_embedding.ori_emb" in new_k:
            continue
        if "token_embedding.add_emb" in new_k:
            continue
        out[new_k] = v

    # Pass 2: token embedding -> use ori_emb directly under the standard name.
    ori_key = "decoder.decoders.token_embedding.ori_emb.weight"
    if ori_key in espnet_state:
        out["decoder.token_embedding.weight"] = espnet_state[ori_key]
    return out


def load_whisper_model_from_espnet(
    espnet_pth: Path,
    whisper_model_size: str,
    device: str,
) -> whisper.model.Whisper:
    """Build a ``whisper.model.Whisper`` of the requested size, then overwrite
    its weights with the converted ESPnet checkpoint.

    ``whisper.load_model`` downloads (and caches under ``~/.cache/whisper/``)
    the pretrained checkpoint for ``whisper_model_size``. The downloaded
    weights are immediately overwritten by our fine-tuned ones via
    ``load_state_dict``. The download is only used to obtain the model
    architecture matching the requested size.
    """
    logger.info(f"Initializing whisper.{whisper_model_size} architecture...")
    model = whisper.load_model(whisper_model_size, device="cpu")
    # NOTE: do not call model.half() here even when fp16=True; whisper's
    # transcribe() casts inputs/cache internally when its own ``fp16`` kwarg
    # is true, and pre-halving the model parameters conflicts with that.

    logger.info(f"Loading ESPnet weights from {espnet_pth}")
    espnet_sd = torch.load(espnet_pth, map_location="cpu", weights_only=False)
    whisper_sd = remap_espnet_to_whisper(espnet_sd)

    missing, unexpected = model.load_state_dict(whisper_sd, strict=False)
    if missing:
        logger.warning(
            f"Whisper model missing {len(missing)} keys "
            f"(first 3: {missing[:3]})"
        )
    if unexpected:
        logger.warning(
            f"Whisper model unexpected {len(unexpected)} keys from ESPnet "
            f"(first 3: {unexpected[:3]})"
        )

    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Text post-processing (match the reference format in data/test/text)
# ---------------------------------------------------------------------------


def postprocess_text(text: str, separator: str = "????") -> str:
    """Rewrite the model's internal speaker separator to ``<sc>`` and collapse
    whitespace, producing transcripts ready for downstream evaluation."""
    text = text.replace(separator, " <sc> ")
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("model_dir", type=Path,
                   help="Directory with model.pth + (optional) config.yaml")
    p.add_argument("--whisper_model", required=True,
                   choices=whisper.available_models(),
                   help="Architecture name for whisper.load_model "
                        "(small, medium, large-v2, ...)")
    p.add_argument("--wav_scp", type=Path, default=Path("data/test/wav.scp"))
    p.add_argument("--out_subdir", default="decode_test_fallback")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fp16", action="store_true",
                   help="Run inference in fp16 (Whisper default).")
    p.add_argument("--separator", default="????",
                   help="SOT separator BPE string emitted by the model.")
    p.add_argument("--language", default="en")
    p.add_argument("--task", default="transcribe",
                   choices=["transcribe", "translate"])
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument(
        "--temperatures",
        default="0.0,0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated temperatures for whisper.transcribe fallback.",
    )
    p.add_argument("--compression_ratio_threshold", type=float, default=2.4)
    p.add_argument("--logprob_threshold", type=float, default=-1.0)
    p.add_argument("--no_speech_threshold", type=float, default=0.6)
    p.add_argument("--max_utts", type=int, default=None)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument(
        "--sot_separator_patch", action="store_true", default=True,
        help="Monkey-patch whisper.decoding.ApplyTimestampRules to allow the "
             "SOT separator (????) after a closing timestamp. Required for "
             "the multi-talker SOT model. Default: True.",
    )
    p.add_argument(
        "--no_sot_separator_patch", dest="sot_separator_patch",
        action="store_false",
        help="Disable the SOT separator patch (default is enabled).",
    )
    args = p.parse_args()

    temperatures = tuple(
        float(x) for x in args.temperatures.split(",") if x.strip()
    )

    out_dir = args.model_dir / args.out_subdir / "1best_recog"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sot_separator_patch:
        _install_sot_separator_patch()
        logger.info(
            f"Patched whisper.decoding.ApplyTimestampRules to permit SOT "
            f"separator (id={WHISPER_SOT_SEPARATOR_ID})"
        )

    t0 = time.time()
    model = load_whisper_model_from_espnet(
        espnet_pth=args.model_dir / "model.pth",
        whisper_model_size=args.whisper_model,
        device=args.device,
    )
    logger.info(f"Model ready in {time.time()-t0:.1f}s")

    # Read utt list
    utts: List[Tuple[str, str]] = []
    with open(args.wav_scp) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utts.append((parts[0], parts[1]))
    if args.max_utts:
        utts = utts[: args.max_utts]
    logger.info(f"Total utterances: {len(utts)}")

    # Tokenizer used to re-decode segment token IDs WITH timestamps preserved
    # (for the text_sot output, which is needed by the DER evaluator).
    sot_tokenizer = whisper.tokenizer.get_tokenizer(
        multilingual=True, task=args.task, language=args.language,
    )

    def _build_text_sot(result: dict) -> str:
        """Re-detokenize each segment's tokens with ``decode_with_timestamps``
        so that ``<|t_start|>``/``<|t_end|>`` markers stay inline. Concatenate
        all segments and rewrite the model's internal separator to ``<sc>``
        so downstream evaluation tooling can split on it cleanly."""
        parts = []
        for seg in result.get("segments", []):
            toks = list(seg.get("tokens", []))
            if toks:
                parts.append(sot_tokenizer.decode_with_timestamps(toks))
        raw = "".join(parts)
        return raw.replace(args.separator, " <sc> ")

    t_start = time.time()
    fail_count = 0
    f_text = open(out_dir / "text", "w")
    f_text_sot = open(out_dir / "text_sot", "w")
    try:
        for i, (uid, path) in enumerate(utts):
            try:
                audio, _ = sf.read(path)
                # whisper.transcribe expects float32 1-D waveform at 16 kHz.
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                result = model.transcribe(
                    audio,
                    language=args.language,
                    task=args.task,
                    temperature=temperatures,
                    compression_ratio_threshold=args.compression_ratio_threshold,
                    logprob_threshold=args.logprob_threshold,
                    no_speech_threshold=args.no_speech_threshold,
                    beam_size=args.beam_size if temperatures[0] == 0.0 else None,
                    fp16=args.fp16,
                    word_timestamps=False,
                    condition_on_previous_text=False,
                    # Disable max_initial_timestamp so the model can emit a
                    # later first timestamp (matches paper-side HF generate()
                    # behaviour). Default 1.0s forces ``<|0.00|>`` even when
                    # actual speech starts >1s into the segment, inflating
                    # false-alarm DER on short/silent-prefix utts.
                    max_initial_timestamp=None,
                )
                text = postprocess_text(
                    result["text"].strip(), separator=args.separator
                )
                f_text.write(f"{uid} {text}\n")
                # text_sot: same content + timestamp markers, for DER eval.
                text_sot = _build_text_sot(result)
                f_text_sot.write(f"{uid} {text_sot}\n")
            except Exception as e:
                fail_count += 1
                logger.warning(f"[{uid}] FAILED: {type(e).__name__}: {e}")
                f_text.write(f"{uid} \n")
                f_text_sot.write(f"{uid} \n")
            if (i + 1) % args.log_every == 0:
                f_text.flush()
                f_text_sot.flush()
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (len(utts) - i - 1) / rate
                logger.info(
                    f"[{i+1}/{len(utts)}] {rate:.2f} utt/s "
                    f"ETA {eta/60:.1f} min"
                )
        f_text.flush()
        f_text_sot.flush()
    finally:
        f_text.close()
        f_text_sot.close()

    total = time.time() - t_start
    logger.info(
        f"Done. {len(utts)} utts in {total/60:.1f} min "
        f"({len(utts)/total:.2f} utt/s).  failures: {fail_count}"
    )


if __name__ == "__main__":
    sys.exit(main())
