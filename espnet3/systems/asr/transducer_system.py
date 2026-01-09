"""ASR system implementation and tokenizer training helpers.

This module adds ASR-specific stages on top of the base system, including
tokenizer training and dataset creation hooks.
"""

import logging
import time
from pathlib import Path

from omegaconf import DictConfig
from espnet3.parallel.parallel import set_parallel
from espnet3.systems.asr.inference import InferenceProvider, TransducerInferenceRunner
from espnet3.systems.asr.system import ASRSystem
from espnet3.systems.base.inference import (_collect_scp_lines, _flatten_results)
from espnet3.systems.base.inference_runner import AbsInferenceRunner

logger = logging.getLogger(__name__)


class ASRTransducerSystem(ASRSystem):
    """ASR Transducer-specific system.

    This system adds:
      - Tokenizer training inside train()
    """
    def infer(self, *args, **kwargs):
        """Run inference on the configured datasets."""
        self._reject_stage_args("infer", args, kwargs)
        logger.info(
            "Inference start | decode_dir=%s",
            getattr(self.infer_config, "decode_dir", None),
        )
        return inference(self.infer_config)


def inference(config: DictConfig):
    """Run inference over all configured test sets and write SCP outputs.

    Args:
        config: Hydra/omegaconf configuration with dataset and decode settings.
    """
    start = time.perf_counter()
    set_parallel(config.parallel)

    test_sets = [test_set.name for test_set in config.dataset.test]
    assert len(test_sets) > 0, "No test set found in dataset"
    assert len(test_sets) == len(set(test_sets)), "Duplicate test key found."

    logger.info(
        "Starting inference | decode_dir=%s test_sets=%s",
        getattr(config, "decode_dir", None),
        test_sets,
    )

    for test_name in test_sets:
        logger.info("===> Processing test set: %s", test_name)
        config.test_set = test_name
        provider = InferenceProvider(config)
        runner = TransducerInferenceRunner(provider=provider, async_mode=False)
        if not isinstance(runner, AbsInferenceRunner):
            raise TypeError(
                f"{type(runner).__name__} must inherit from AbsInferenceRunner"
            )
        dataset_length = len(provider.build_dataset(config))
        logger.info("===> Processing %d samples..", dataset_length)
        out = runner(list(range(dataset_length)))
        if out is None:
            raise RuntimeError("Async inference is not supported in this entrypoint.")
        results = _flatten_results(out)
        scp_lines = _collect_scp_lines(
            results,
            idx_key=runner.idx_key,
            hyp_keys=runner.hyp_key,
            ref_keys=runner.ref_key,
        )

        # create scp files
        output_dir = Path(config.decode_dir) / test_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for key, lines in scp_lines.items():
            with open(output_dir / f"{key}.scp", "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        logger.info(
            "Finished test set %s | outputs=%s",
            test_name,
            output_dir,
        )

    logger.info("Inference finished in %.2fs", time.perf_counter() - start)
