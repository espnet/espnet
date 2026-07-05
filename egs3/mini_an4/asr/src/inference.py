"""Inference helpers for Mini AN4 ASR recipes."""

from __future__ import annotations

from espnet2.bin.asr_transducer_inference import Speech2Text as TransducerSpeech2Text


class TransducerInferenceWrapper(TransducerSpeech2Text):
    """Wrap ESPnet2 transducer inference for ESPnet3 recipe output functions.

    This wrapper is intended for ``config.model._target_`` in the Mini AN4
    transducer recipe. It keeps the original ESPnet2 constructor and decoding
    behavior, but converts the raw N-best hypotheses into the tuple results
    returned by ``hypotheses_to_results()`` so recipe-local ``output_fn`` code
    can read plain text directly.

    Returns:
        list[tuple[object, object, object, object]]: N-best tuples in the same
        format as ``Speech2Text.hypotheses_to_results()``.

    Example:
        model:
          _target_: egs3.mini_an4.asr.src.inference.TransducerInferenceWrapper
          asr_train_config: ${exp_dir}/config.yaml
          asr_model_file: ${exp_dir}/last.ckpt
          beam_size: 1
    """

    def __call__(self, speech):
        """Run transducer inference and convert hypotheses to tuple results."""
        return self.hypotheses_to_results(super().__call__(speech))


def build_output(data, model_output, idx):
    """Build a dict of outputs for SCP writing."""
    utt_id = data.get("utt_id", str(idx))
    hyp = model_output[0][0]
    ref = data.get("text", "")
    return {"utt_id": utt_id, "hyp": hyp, "ref": ref}


def build_output_transducer(data, model_output, idx):
    """Build a dict of outputs for transducer models."""
    utt_id = data.get("utt_id", str(idx))
    hyp = model_output[0][0]
    ref = data.get("text", "")
    return {"utt_id": utt_id, "hyp": hyp, "ref": ref}
