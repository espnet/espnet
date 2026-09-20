"""Average the final retained checkpoints before VOiCES inference."""

from pathlib import Path

import torch

from espnet2.torch_utils.safe_torch_load import safe_torch_load


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
            build_output(sample, output, index)
            for sample, output, index in zip(data, model_output, idx, strict=True)
        ]
    return {"utt_id": str(idx), "hyp": model_output[0][0] or "", "ref": data["text"]}


def average_checkpoints(checkpoint_dir, max_checkpoints=10):
    """Average retained valid/acc checkpoints, including the last training epoch.

    Upstream averaging runs before ModelCheckpoint at validation end, so its
    alias can lag by one epoch. Read retained weights plus exact valid/acc
    scores from the final full checkpoint. Sum in metric-ranked order, with
    earlier epochs first for ties. If fewer than max_checkpoints exist, use
    the best one, matching ESPnet2. Integer buffers are summed.
    Experiments must use separate output directories.

    Args:
        checkpoint_dir: Completed Lightning experiment directory.
        max_checkpoints: Expected upper bound of retained best checkpoints.

    Returns:
        Path to an averaged plain model state dictionary.

    Raises:
        RuntimeError: No checkpoints, too many checkpoints, or incompatible keys.
    """
    directory = Path(checkpoint_dir)
    paths = sorted(directory.glob("epoch*_step*_valid.acc.ckpt"))
    if not 0 < len(paths) <= max_checkpoints:
        raise RuntimeError(
            f"Expected 1..{max_checkpoints} retained checkpoints: {paths}"
        )
    if len(paths) > 1:
        # Best-model files contain weights only. The final full checkpoint
        # retains ModelCheckpoint's exact scores, including the last epoch.
        full_paths = list(directory.glob("step*.ckpt"))
        if not full_paths:
            raise RuntimeError("Missing final checkpoint with valid/acc scores")
        latest = max(full_paths, key=lambda path: int(path.stem.removeprefix("step")))
        checkpoint = safe_torch_load(latest, map_location="cpu")
        scores = {}
        for state in checkpoint.get("callbacks", {}).values():
            if state.get("monitor") == "valid/acc":
                scores.update(
                    {
                        Path(path).name: float(score)
                        for path, score in state["best_k_models"].items()
                    }
                )
        if any(path.name not in scores for path in paths):
            raise RuntimeError("Final checkpoint lacks scores for retained models")
        paths.sort(
            key=lambda path: (
                -scores[path.name],
                int(path.name.split("_")[0].removeprefix("epoch")),
            )
        )
    # Native average_nbest_models falls back to the best one if nbest=10
    # cannot be satisfied; it does not average the available smaller set.
    if len(paths) < max_checkpoints:
        paths = paths[:1]
    averaged = None
    for path in paths:
        state = safe_torch_load(path, map_location="cpu")["state_dict"]
        if averaged is None:
            averaged = state
        else:
            if averaged.keys() != state.keys():
                raise RuntimeError(f"Incompatible checkpoint keys: {path}")
            for key in averaged:
                averaged[key] += state[key]
    for key, value in averaged.items():
        if value.is_floating_point() or value.is_complex():
            averaged[key] = value / len(paths)
    output = directory / "valid.acc.final_ave.pth"
    temporary = output.with_suffix(".tmp")
    torch.save(averaged, temporary)
    temporary.replace(output)
    return output


def load_averaged_model(checkpoint_dir, asr_train_config, max_checkpoints=10, **kwargs):
    """Load retained ASR weights and optional LM through Speech2Text.

    Args:
        checkpoint_dir: Completed Lightning training directory.
        asr_train_config: Native ASR config saved by the train stage.
        max_checkpoints: Maximum number of retained checkpoints to average.
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
        asr_model_file=str(average_checkpoints(checkpoint_dir, max_checkpoints)),
        **kwargs,
    )
