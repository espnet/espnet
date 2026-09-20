"""Checkpoint selection and inference output formatting for AN4."""

from pathlib import Path


def build_output(data, model_output, idx):
    """Pair inference hypotheses with reference text and manifest indices.

    Args:
        data: One dataset sample, or a list of samples in batch mode.
        model_output: Speech2Text n-best output, or matching batched outputs.
        idx: Manifest index, or the corresponding list of indices.

    Returns:
        A dictionary (or list) containing utt_id, hyp and ref. Empty
        predictions remain empty so the scorer counts deletions correctly.

    Raises:
        ValueError: Batched inputs and outputs have different lengths.
    """
    if isinstance(data, list):
        return [
            build_output(d, output, index)
            for d, output, index in zip(data, model_output, idx, strict=True)
        ]
    return {"utt_id": str(idx), "hyp": model_output[0][0] or "", "ref": data["text"]}


def find_best_checkpoint(checkpoint_dir):
    """Find the single checkpoint retained by the valid/acc top-one callback.

    Refuse ambiguous directories instead of silently selecting weights from
    another run. Upstream checkpoint averaging may run before ModelCheckpoint,
    so its averaged alias is not reliable for a one-epoch smoke test.
    """
    paths = sorted(Path(checkpoint_dir).glob("epoch*_step*_valid.acc.ckpt"))
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected one best valid/acc checkpoint in {checkpoint_dir}, "
            f"found {len(paths)}. Train in a separate experiment directory."
        )
    return paths[0]


def load_best_model(checkpoint_dir, asr_train_config, **kwargs):
    """Load retained ASR weights and optional LM through Speech2Text.

    Args:
        checkpoint_dir: Completed Lightning training directory.
        asr_train_config: Native ASR config saved by the train stage.
        **kwargs: Speech2Text options, including lm_file, lm_train_config,
            lm_weight, beam_size and device.

    Returns:
        Ready-to-call Speech2Text model.

    Raises:
        RuntimeError: The retained checkpoint set is missing or ambiguous.
        FileNotFoundError: An ASR/LM config or checkpoint does not exist.
    """
    from espnet2.bin.asr_inference import Speech2Text

    return Speech2Text(
        asr_train_config=asr_train_config,
        asr_model_file=str(find_best_checkpoint(checkpoint_dir)),
        **kwargs,
    )
