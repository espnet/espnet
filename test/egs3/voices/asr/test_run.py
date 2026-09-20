"""Run the shared stage entrypoint with the VOiCES system and a CPU fixture."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from espnet3.utils.config_utils import load_and_merge_config

RECIPE = Path(__file__).resolve().parents[4] / "egs3/voices/asr"


@pytest.mark.execution_timeout(120)
def test_complete_stage_pipeline_on_fixture(corpus, monkeypatch):
    """Exercise all six shared stages with a tiny CPU model and real WAV files."""
    from egs3.TEMPLATE.asr.run import DEFAULT_STAGES, build_parser, main
    from egs3.voices.asr.src import tokenizer as tokenizer_module
    from egs3.voices.asr.src.system import VoicesSystem

    recipe, source = corpus
    monkeypatch.setattr(
        tokenizer_module,
        "gather_training_text",
        lambda **_: [" ".join("ABCDEFGHIJKLMNOPQRSTUVWXYZ")] * 5,
    )
    training = load_and_merge_config(
        RECIPE / "conf/training_devkit.yaml", "training.yaml", resolve=False
    )
    training.recipe_dir = str(RECIPE)
    training.data_dir = str(recipe / "data")
    training.exp_dir = str(recipe / "experiment")
    training.stats_dir = str(recipe / "statistics")
    training.create_dataset = {"recipe_dir": str(recipe), "source_dir": str(source)}
    for split in ("train", "valid"):
        training.dataset[split][0].data_src = "egs3.voices.asr.dataset"
        training.dataset[split][0].data_src_args.recipe_dir = str(recipe)
        training.dataloader[split].iter_factory.num_workers = 0
    training.tokenizer.vocab_size = 30
    training.tokenizer.save_path = str(recipe / "tokenizer")
    training.model.encoder_conf.update(
        output_size=16, attention_heads=2, linear_units=32, num_blocks=1
    )
    training.model.decoder_conf.update(attention_heads=2, linear_units=32, num_blocks=1)
    training.num_device = 1
    training.espnet2_compat.accum_grad = 1
    training.trainer.update(
        accelerator="cpu",
        devices=1,
        # This single-process fixture must not leave a DDP group in pytest.
        strategy="auto",
        precision="32-true",
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        accumulate_grad_batches=1,
    )
    training.dataloader.train.iter_factory.batches.batch_bins = 100000
    inference = load_and_merge_config(
        RECIPE / "conf/inference.yaml", "inference.yaml", resolve=False
    )
    inference.recipe_dir = str(RECIPE)
    inference.exp_dir = str(recipe / "experiment")
    inference.inference_dir = str(recipe / "inference")
    inference.batch_size = 1
    inference.model.update(
        beam_size=2, maxlenratio=0.1, lm_weight=0.0, lm_file=None, lm_train_config=None
    )
    for entry in inference.dataset.test:
        entry.data_src = "egs3.voices.asr.dataset"
        entry.data_src_args.recipe_dir = str(recipe)
        entry.data_src_args.limit = 1
    metrics = load_and_merge_config(
        RECIPE / "conf/metrics.yaml", "metrics.yaml", resolve=False
    )
    metrics.metrics[2].metric.bpemodel = str(recipe / "tokenizer/unigram.model")
    metrics_path = recipe / "metrics.yaml"
    OmegaConf.save(metrics, metrics_path)
    training_path = recipe / "training.yaml"
    inference_path = recipe / "inference.yaml"
    OmegaConf.save(training, training_path)
    OmegaConf.save(inference, inference_path)
    args = build_parser(DEFAULT_STAGES).parse_args(
        [
            "--stages",
            "create_dataset",
            "train_tokenizer",
            "collect_stats",
            "train",
            "infer",
            "measure",
            "--training_config",
            str(training_path),
            "--inference_config",
            str(inference_path),
            "--metrics_config",
            str(metrics_path),
        ]
    )
    main(args, VoicesSystem, DEFAULT_STAGES)
    assert (recipe / "experiment/valid.acc.final_ave.pth").is_file()
    assert (recipe / "inference/test/hyp.scp").is_file()
    assert (recipe / "inference/metrics.json").is_file()
