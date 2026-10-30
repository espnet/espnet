#!/usr/bin/env python3
"""LibriCSS benchmark recipe runner.

LibriCSS is an evaluation-only benchmark: no model is trained here. The ASR
acoustic model and the speaker embedding extractor are both pretrained. The
stage flow is therefore::

    create_dataset -> segment -> diarize -> infer -> measure

where ``segment`` and ``diarize`` are custom stages added by this recipe on
top of ``ASRSystem``. ``LibriCSSSystem.stages`` declares exactly this stage
list (the training-related ``ASRSystem`` stages are dropped), and
``launch()`` dispatches each stage to the method of the same name.

Quick usage (diarized flow, webrtcvad segmentation + NME spectral
clustering, then SA-WER scoring)::

    python run.py --eval_config conf/eval.yaml \
        --inference_config conf/inference.yaml \
        --metrics_config conf/metrics.yaml --stages all

Oracle-segment flow (skip ``diarize``, score plain per-utterance WER)::

    python run.py --eval_config conf/eval_oracle.yaml \
        --inference_config conf/inference_oracle.yaml \
        --metrics_config conf/metrics_oracle.yaml \
        --stages create_dataset segment infer measure

``--eval_config`` is this recipe's name for ``--training_config`` (a pure
alias: both flags feed the same framework slot) -- the recipe trains
nothing, so the config it points at (``conf/eval.yaml``) carries only
dataset-creation settings and the parameters of the recipe-local stages.
"""

import shutil
from pathlib import Path
from typing import ClassVar

from src.diarization import run_diarization
from src.segmentation import run_segmentation

from espnet3.contract.stages import StageSpec
from espnet3.systems.esp2_asr.system import ASRSystem
from espnet3.systems.launch import build_parser, launch


class LibriCSSSystem(ASRSystem):
    """ASRSystem with the LibriCSS ``segment`` and ``diarize`` stages.

    Benchmark only: the tokenizer/training/publication stages of
    ``ASRSystem`` are not declared. Both custom stages read their settings
    from the ``libricss`` block of ``conf/eval.yaml``, which the framework
    carries in its ``training_config`` slot, so the recipe keeps a single
    place to configure data preparation, segmentation, and diarization.
    """

    stages: ClassVar[tuple[StageSpec, ...]] = (
        StageSpec(name="create_dataset", config="training", log_dir="data_dir"),
        StageSpec(name="segment", config="training", log_dir="exp_dir"),
        StageSpec(name="diarize", config="training", log_dir="exp_dir"),
        StageSpec(name="infer", config="inference", log_dir="inference_dir"),
        StageSpec(name="measure", config="metrics", log_dir="inference_dir"),
    )

    def segment(self, *args, **kwargs):
        """Write per-recording speech segment manifests.

        Port of egs/libri_css/asr1/local/segment.sh (webrtcvad) and
        local/segment_diarize_oracle.sh (oracle segments). Output:
        ``${exp_dir}/segments/<split>/<reco>.json``.
        """
        self._reject_stage_args("segment", args, kwargs)
        run_segmentation(self.training_config)

    def diarize(self, *args, **kwargs):
        """Cluster subsegment speaker embeddings into flat speaker turns.

        Port of egs/libri_css/asr1/local/diarize.sh with the default
        ``diarizer_type=spectral``: x-vector-style embeddings, per-recording
        mean-centered cosine similarity, NME spectral clustering, and
        midpoint overlap cutting. Output:
        ``${exp_dir}/diarized/<split>/<reco>.json`` (+ optional RTTM).
        """
        self._reject_stage_args("diarize", args, kwargs)
        run_diarization(self.training_config)

    def infer(self, *args, **kwargs):
        """Decode the manifest segments, dropping stale scoring caches first.

        ``measure`` caches the ``dataset:<column>`` inputs it reads (speaker
        labels, recordings, start times, oracle transcripts) under
        ``${inference_dir}/<test_name>/dataset/``. A new ``infer`` run may
        decode different segments (e.g. after re-running ``segment`` or
        ``diarize``), so those caches are removed before decoding.
        """
        self._reject_stage_args("infer", args, kwargs)
        inference_dir = Path(self.inference_config.inference_dir)
        for test_set in self.inference_config.dataset.test:
            shutil.rmtree(inference_dir / test_set.name / "dataset", ignore_errors=True)
        return super().infer()


if __name__ == "__main__":
    parser = build_parser(LibriCSSSystem)
    # Evaluation-only recipe: expose the --training_config slot under the
    # more meaningful --eval_config name. Same argparse dest, so both flags
    # work.
    parser.add_argument(
        "--eval_config",
        dest="training_config",
        metavar="EVAL_CONFIG",
        default=None,
        type=Path,
        help=(
            "Recipe eval config: dataset creation + segment/diarize stage "
            "parameters (alias of --training_config; nothing is trained)."
        ),
    )
    # LibriCSSSystem lives in this recipe, not under espnet3.systems.<x>, so
    # the package holding the default configs cannot be derived from it.
    launch(LibriCSSSystem, conf_package="egs3.TEMPLATE.esp2_asr", parser=parser)
