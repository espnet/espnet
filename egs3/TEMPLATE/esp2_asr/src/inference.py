"""Format native ASR results for the recipe's SCP writers."""


def build_output(*, data, model_output, idx):
    """Return one text record per input, retaining the dataset identifier."""
    if isinstance(data, list):
        return [
            build_output(data=item, model_output=output, idx=i)
            for item, output, i in zip(data, model_output, idx, strict=True)
        ]
    best = model_output[0]
    if isinstance(best, list):
        best = best[0]
    return {"utt_id": data.get("utt_id", str(idx)), "text": best[0] or ""}
