"""ASR-specific output formatting for the shared inference runner."""

from __future__ import annotations

from espnet3.systems.base.inference_runner import InferenceRunner


class ASRInferenceRunner(InferenceRunner):
    """Convert ESPnet2 ASR n-best results to the contracted ``text`` field."""

    @staticmethod
    def _format_one(hypotheses):
        """Return the best hypothesis as a named text field."""
        best = hypotheses[0]
        if isinstance(best, list):
            best = best[0]
        return {"text": best[0] or ""}

    @staticmethod
    def forward(idx, dataset=None, model=None, **kwargs):
        """Run ESPnet2 decoding and format its best text hypothesis."""
        result = InferenceRunner.forward(idx, dataset, model, **kwargs)
        if isinstance(idx, (list, tuple)):
            return [ASRInferenceRunner._format_one(item) for item in result]
        return ASRInferenceRunner._format_one(result)
