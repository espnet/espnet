"""Classification output formatting for the shared inference runner."""

from __future__ import annotations

from espnet3.systems.base.inference_runner import InferenceRunner


class ESP2CLSInferenceRunner(InferenceRunner):
    """Convert ESPnet2 ``Classification`` results to ``hyp`` and ``score``."""

    @staticmethod
    def prepare_model_inputs(inputs):
        """Pass the checked audio to ``Classification`` as a plain array.

        Args:
            inputs: The backend's keyword arguments; ``speech`` holds an
                ``Audio``.

        Returns:
            A copy of ``inputs`` whose ``speech`` is the ``Audio``'s samples.
        """
        return {**inputs, "speech": inputs["speech"].array}

    @staticmethod
    def format_output(result, *, batched):
        """Name the predicted label and the per-class scores.

        Args:
            result: The ``(prediction, scores, hyp)`` tuple of
                ``Classification``.
            batched: Whether the call was a batch.

        Returns:
            ``{"hyp": <label>, "score": <space-separated class scores>}``.

        Raises:
            ValueError: If the call was a batch.

        Examples:
            >>> import torch
            >>> result = ([1], torch.tensor([0.25, 0.75]), "joy")
            >>> ESP2CLSInferenceRunner.format_output(result, batched=False)
            {'hyp': 'joy', 'score': '0.25 0.75'}
        """
        if batched:
            raise ValueError("Use batch_size: null for native Classification")
        _, scores, hyp = result
        return {"hyp": hyp, "score": " ".join(str(value) for value in scores.tolist())}
