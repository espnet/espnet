import logging
import time
from pathlib import Path

from omegaconf import DictConfig

from espnet3.parallel.parallel import set_parallel
from espnet3.systems.asr.inference import InferenceProvider, InferenceRunner

logger = logging.getLogger(__name__)


def inference(config: DictConfig):
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
        runner = InferenceRunner(
            provider=provider,
            async_mode=False,
        )
        dataset_length = len(provider.build_dataset(config))
        logger.info("===> Processing %d samples..", dataset_length)
        out = runner(list(range(dataset_length)))

        # create scp files
        (Path(config.decode_dir) / test_name).mkdir(parents=True, exist_ok=True)
        with open(Path(config.decode_dir) / test_name / "ref.scp", "w") as f:
            f.write("\n".join([f"{result['idx']} {result['ref']}" for result in out]))

        with open(Path(config.decode_dir) / test_name / "hyp.scp", "w") as f:
            f.write("\n".join([f"{result['idx']} {result['hyp']}" for result in out]))
        logger.info(
            "Finished test set %s | outputs=%s",
            test_name,
            Path(config.decode_dir) / test_name,
        )

    logger.info("Inference finished in %.2fs", time.perf_counter() - start)
