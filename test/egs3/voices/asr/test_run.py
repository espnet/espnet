"""Exercise the stock ASR stages and native LM using a small CPU corpus."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from espnet3.utils.config_utils import load_and_merge_config

RECIPE = Path(__file__).resolve().parents[4] / "egs3/voices/asr"


@pytest.mark.execution_timeout(180)
def test_complete_stage_pipeline_on_fixture(corpus, monkeypatch):
    """Train, load and score ASR with a nonzero trained LM fusion weight."""
    import json

    import yaml

    from egs3.TEMPLATE.asr.run import DEFAULT_STAGES, build_parser, main
    from egs3.voices.asr.src import tokenizer as tokenizer_module
    from egs3.voices.asr.src.language_model import train_language_model
    from espnet3.systems.asr.system import ASRSystem

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
    training.dataloader.train.iter_factory.batches.batch_bins = 100000
    training.num_device = 1
    training.best_model_criterion = [["valid/acc", 1, "max"]]
    training.trainer.update(
        accelerator="cpu",
        devices=1,
        strategy="auto",
        precision="32-true",
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        accumulate_grad_batches=1,
    )
    inference = load_and_merge_config(
        RECIPE / "conf/inference.yaml", "inference.yaml", resolve=False
    )
    inference.recipe_dir = str(RECIPE)
    inference.exp_dir = str(recipe / "experiment")
    inference.inference_dir = str(recipe / "inference")
    inference.batch_size = 1
    inference.model.update(
        asr_model_file=str(recipe / "experiment/valid.acc.ave_1best.pth"),
        beam_size=2,
        maxlenratio=0.1,
        lm_train_config=str(recipe / "lm/config.yaml"),
        lm_file=str(recipe / "lm/valid.loss.ave.pth"),
    )
    for entry in inference.dataset.test:
        entry.data_src = "egs3.voices.asr.dataset"
        entry.data_src_args.recipe_dir = str(recipe)
        entry.data_src_args.limit = 1
    metrics = load_and_merge_config(
        RECIPE / "conf/metrics.yaml", "metrics.yaml", resolve=False
    )
    metrics.metrics[2].metric.bpemodel = str(recipe / "tokenizer/unigram.model")
    configs = {"training": training, "inference": inference, "metrics": metrics}
    for name, config in configs.items():
        OmegaConf.save(config, recipe / f"{name}.yaml")

    def run(stages):
        argv = ["--stages", *stages]
        for name in configs:
            argv += [f"--{name}_config", str(recipe / f"{name}.yaml")]
        main(build_parser(DEFAULT_STAGES).parse_args(argv), ASRSystem, DEFAULT_STAGES)

    # Reload YAML for train: stock collect_stats removes normalization in memory.
    run(["create_dataset", "train_tokenizer", "collect_stats"])
    assert (recipe / "statistics/train/feats_shape").is_file()
    run(["train"])
    assert (
        yaml.safe_load((recipe / "experiment/config.yaml").read_text())["normalize"]
        == "global_mvn"
    )
    lm = OmegaConf.create(
        {
            "exp_dir": str(recipe / "lm"),
            "native_config": str(RECIPE / "conf/lm_native.yaml"),
            "tokenizer_dir": str(recipe / "tokenizer"),
            "train_text": str(recipe / "data/lm/train.txt"),
            "valid_text": str(recipe / "data/lm/valid.txt"),
            "test_text": str(recipe / "data/lm/test.txt"),
            "ngpu": 0,
            "native_overrides": {
                "lm": "seq_rnn",
                "lm_conf": {"nlayers": 1, "unit": 16},
                "max_epoch": 1,
                "batch_type": "folded",
                "batch_size": 2,
                "optim": "adam",
                "optim_conf": {"lr": 0.001},
                "scheduler": None,
                "scheduler_conf": {},
                "num_workers": 0,
            },
        }
    )
    # Replace the model-specific config rather than merging Transformer options.
    tiny_lm = recipe / "tiny_lm.yaml"
    OmegaConf.save(OmegaConf.create({}), tiny_lm)
    lm.native_config = str(tiny_lm)
    train_language_model(lm)
    run(["infer", "measure"])
    assert (recipe / "experiment/valid.acc.ave_1best.pth").is_file()
    assert (recipe / "inference/test/hyp.scp").is_file()
    scores = json.loads((recipe / "inference/metrics.json").read_text())
    assert scores
