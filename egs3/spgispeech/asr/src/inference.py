"""Inference output formatting for the SPGISpeech ASR recipe.

`conf/inference.yaml` points `output_fn` here. ESPnet3 calls it once per sample
and writes one SCP file per returned key under
`${inference_dir}/<test_name>/`, so the keys below become `hyp.scp` and
`ref.scp` -- the two files `conf/metrics.yaml` scores.
"""


def build_output(data, model_output, idx):
    """Turn one Speech2Text result into the dict ESPnet3 writes to SCP.

    Args:
        data: The raw dataset sample. `SPGISpeechDataset.__getitem__` returns
            only `speech` and `text` -- see below on `utt_id`. When the
            inference config sets `batch_size`, this is instead a list of
            samples -- see NOTE ON BATCHING.
        model_output: `espnet2.bin.asr_inference.Speech2Text.__call__` output,
            an n-best list of `(text, token, token_int, hypothesis)` tuples, so
            `[0][0]` is the best hypothesis text. A list of these when `data`
            is a list.
        idx: Index of the sample within its test set. A list of indices when
            `data` is a list.

    Returns:
        dict with `utt_id`, `hyp` and `ref`, or a list of such dicts.

    NOTE ON BATCHING. `batch_size` in `conf/inference.yaml` (inherited from
    egs3/TEMPLATE/asr/conf/inference.yaml's default) makes ESPnet3 group
    consecutive dataset indices and call this function once per group, with
    `data`, `model_output` and `idx` all lists of matching length -- see
    egs3/TEMPLATE/asr/src/inference.py:build_output, which this mirrors.

    NOTE ON utt_id. SPGISpeech utterance ids are deliberately NOT in the
    dataset sample: espnet2's CommonPreprocessor is @typechecked as returning
    Dict[str, np.ndarray] and passes unknown keys through unchanged, so a string
    `utt_id` aborts collect_stats and training with
        TypeCheckError: value of key 'utt_id' of the return value (dict)
        is not an instance of numpy.ndarray
    (see the comment in dataset/dataset.py:__getitem__). The sample index is
    used instead, exactly as egs3/librispeech_100/asr/src/inference.py does.
    Scoring is unaffected -- `ref` and `hyp` are emitted from one pass over one
    ordering, so the two SCPs line up key for key -- but the SCP keys are
    integers rather than SPGISpeech ids, which makes hand-inspection harder.
    """
    if isinstance(data, list):
        return [build_output(d, o, i) for d, o, i in zip(data, model_output, idx)]
    utt_id = data.get("utt_id", str(idx))
    hyp = model_output[0][0]
    ref = data.get("text", "")
    return {"utt_id": utt_id, "hyp": hyp, "ref": ref}
