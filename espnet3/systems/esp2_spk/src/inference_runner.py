"""Speaker-verification input and output adaptation for the shared runner."""

from __future__ import annotations

from espnet3.systems.base.inference_runner import InferenceRunner


class ESP2SPKInferenceRunner(InferenceRunner):
    """Adapt :class:`~espnet3.systems.esp2_spk.src.speech2score.TrialScorer`.

    The scorer takes the two raw waveforms of one trial and returns a bare
    float, so the runner hands it the sample arrays and names the result
    ``score``. ``forward`` is inherited, keeping the shared contract checks
    and sample IDs.
    """

    @staticmethod
    def prepare_model_inputs(inputs):
        """Hand the scorer the waveforms, rejecting a batch.

        Args:
            inputs: ``{"speech": Audio, "speech2": Audio}``, already checked
                as mono audio at the contract rate.

        Returns:
            The same keys holding the 1-D sample arrays.

        Raises:
            ValueError: If the stage was configured with a batch size; a
                trial is scored one pair per call.

        Examples:
            >>> import numpy as np
            >>> from espnet3.contract.kinds.audio import Audio
            >>> a = Audio(np.zeros(8000, dtype=np.float32), 16000)
            >>> out = ESP2SPKInferenceRunner.prepare_model_inputs(
            ...     {"speech": a, "speech2": a}
            ... )
            >>> out["speech"].shape
            (8000,)
        """
        if any(isinstance(value, list) for value in inputs.values()):
            raise ValueError("Use batch_size: null for trial scoring")
        return {name: value.array for name, value in inputs.items()}

    @staticmethod
    def format_output(result, *, batched):
        """Name the scorer's float ``score``.

        Args:
            result: The similarity score of one trial.
            batched: Whether the call was a batch; never true here, since
                :meth:`prepare_model_inputs` refuses one.

        Returns:
            ``{"score": float}``.

        Examples:
            >>> ESP2SPKInferenceRunner.format_output(0.5, batched=False)
            {'score': 0.5}
        """
        if batched:
            raise ValueError("Use batch_size: null for trial scoring")
        return {"score": float(result)}
