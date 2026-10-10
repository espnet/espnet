#!/usr/bin/env python3
"""LibriCSS benchmark recipe runner.

LibriCSS is an evaluation-only benchmark: no model is trained here. The ASR
acoustic model and the speaker embedding extractor are both pretrained. The
stage flow is therefore::

    create_dataset -> segment -> diarize -> infer -> measure

where ``segment`` and ``diarize`` are custom stages added by this recipe on
top of the ESPnet3 template system. They are plain methods on
``LibriCSSSystem``, dispatched by ``run_stages`` via ``getattr``; no framework
change is needed.

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

``--eval_config`` is this recipe's name for the template's
``--training_config`` (a pure alias: both flags feed the same framework
slot) -- the recipe trains nothing, so the config it points at
(``conf/eval.yaml``) carries only dataset-creation settings and the
parameters of the recipe-local stages.
"""

from pathlib import Path

from src.diarization import run_diarization
from src.segmentation import run_segmentation

from egs3.TEMPLATE.asr.run import build_parser, main, parse_cli_and_stage_args
from espnet3.systems.asr.system import ASRSystem

# LibriCSS extends the template stage list with two recipe-local stages.
# The template's train/collect_stats/pack stages are not used (benchmark only).
STAGES = [
    "create_dataset",
    "segment",
    "diarize",
    "infer",
    "measure",
]


class LibriCSSSystem(ASRSystem):
    """ASRSystem with the LibriCSS ``segment`` and ``diarize`` stages.

    Both custom stages read their settings from the ``libricss`` block of
    ``conf/eval.yaml``, which the framework carries in its
    ``training_config`` slot (template contract), so the recipe keeps a
    single place to configure data preparation, segmentation, and
    diarization.
    """

    def _require_eval_config(self, stage: str):
        if self.training_config is None:
            raise RuntimeError(
                f"The '{stage}' stage requires the eval config. "
                "Pass --eval_config conf/eval.yaml."
            )

    def segment(self):
        """Write per-recording speech segment manifests.

        Port of egs/libri_css/asr1/local/segment.sh (webrtcvad) and
        local/segment_diarize_oracle.sh (oracle segments). Output:
        ``${exp_dir}/segments/<split>/<reco>.json``.
        """
        self._require_eval_config("segment")
        run_segmentation(self.training_config)

    def diarize(self):
        """Cluster subsegment speaker embeddings into flat speaker turns.

        Port of egs/libri_css/asr1/local/diarize.sh with the default
        ``diarizer_type=spectral``: x-vector-style embeddings, per-recording
        mean-centered cosine similarity, NME spectral clustering, and
        midpoint overlap cutting. Output:
        ``${exp_dir}/diarized/<split>/<reco>.json`` (+ optional RTTM).
        """
        self._require_eval_config("diarize")
        run_diarization(self.training_config)


if __name__ == "__main__":
    parser = build_parser(stages=STAGES)
    # Evaluation-only recipe: expose the template's --training_config slot
    # under the more meaningful --eval_config name. Same argparse dest, so
    # both flags work and main() is reused unchanged.
    parser.add_argument(
        "--eval_config",
        dest="training_config",
        default=None,
        type=Path,
        help=(
            "Recipe eval config: dataset creation + segment/diarize stage "
            "parameters (alias of --training_config; nothing is trained)."
        ),
    )
    args, stages_to_run = parse_cli_and_stage_args(parser, stages=STAGES)
    main(
        args=args,
        system_cls=LibriCSSSystem,
        stages=stages_to_run,
    )
