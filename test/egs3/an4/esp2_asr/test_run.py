"""Exercise the stock ASR and LM stages using a small CPU corpus."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from espnet3.utils.config_utils import load_and_merge_config

RECIPE = Path(__file__).resolve().parents[4] / "egs3/an4/esp2_asr"


@pytest.mark.execution_timeout(180)
def test_complete_stage_pipeline_on_fixture(corpus, monkeypatch):
    """Train, load and score ASR with a nonzero trained LM fusion weight."""
    import json

    import yaml

    from egs3.an4.esp2_asr.src import tokenizer as tokenizer_module
    from egs3.an4.esp2_asr.src.language_model import prepare_lm_text
    from egs3.TEMPLATE.esp2_asr.run import DEFAULT_STAGES, build_parser, main
    from espnet3.systems.esp2_asr.lm_system import LMSystem
    from espnet3.systems.esp2_asr.system import ASRSystem

    recipe = corpus
    source = corpus / "downloads/an4"
    monkeypatch.setattr(
        tokenizer_module,
        "gather_training_text",
        lambda **_: [" ".join("ABCDEFGHIJKLMNOPQRSTUVWXYZ")] * 5,
    )
    training = load_and_merge_config(
        RECIPE / "conf/training_sinc_rnn.yaml", "training.yaml", resolve=False
    )
    training.recipe_dir = str(RECIPE)
    training.data_dir = str(recipe / "data")
    training.exp_dir = str(recipe / "experiment")
    training.stats_dir = str(recipe / "statistics")
    training.create_dataset = {
        "recipe_dir": str(recipe),
        "source_dir": str(source),
        "dev_size": 1,
    }
    for split in ("train", "valid"):
        training.dataset[split][0].data_src = "egs3.an4.esp2_asr.dataset"
        training.dataset[split][0].data_src_args.recipe_dir = str(recipe)
        training.dataloader[split].iter_factory.num_workers = 0
    training.tokenizer.vocab_size = 30
    training.tokenizer.save_path = str(recipe / "tokenizer")
    training.model.encoder_conf.update(
        num_layers=1, hidden_size=16, output_size=16, subsample=[1]
    )
    training.model.decoder_conf.hidden_size = 16
    training.dataloader.train.iter_factory.batches.batch_size = 2
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
        accumulate_grad_batches=1,
    )
    inference = load_and_merge_config(
        RECIPE / "conf/inference.yaml", "inference.yaml", resolve=False
    )
    inference.recipe_dir = str(RECIPE)
    inference.exp_dir = str(recipe / "experiment")
    inference.inference_dir = str(recipe / "inference")
    inference.batch_size = None
    inference.model.update(
        asr_model_file=str(recipe / "experiment/valid.acc.ave_1best.pth"),
        beam_size=2,
        maxlenratio=0.1,
        lm_train_config=str(recipe / "lm/config.yaml"),
        lm_file=str(recipe / "lm/valid.loss.ave_1best.pth"),
    )
    for entry in inference.dataset.test:
        entry.data_src = "egs3.an4.esp2_asr.dataset"
        entry.data_src_args.recipe_dir = str(recipe)
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
    lm = load_and_merge_config(
        RECIPE / "conf/training_lm.yaml", "training.yaml", resolve=False
    )
    lm.recipe_dir = str(RECIPE)
    lm.data_dir = str(recipe / "data")
    lm.exp_dir = str(recipe / "lm")
    lm.tokenizer_dir = str(recipe / "tokenizer")
    lm.external_text = None
    lm.num_device = 1
    lm.model = {
        "token_list": "${tokenizer_dir}/tokens.txt",
        "lm": "seq_rnn",
        "lm_conf": {"nlayers": 1, "unit": 16},
    }
    lm.optimizer.lr = 0.001
    lm.scheduler = "${constant_scheduler}"
    lm.constant_scheduler = {
        "_target_": "torch.optim.lr_scheduler.ConstantLR",
        "factor": 1.0,
        "total_iters": 1,
    }
    lm.best_model_criterion = [["valid/loss", 1, "min"]]
    lm.trainer.update(
        accelerator="cpu",
        devices=1,
        strategy="auto",
        precision="32-true",
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=1,
        accumulate_grad_batches=1,
    )
    lm.dataloader.train.iter_factory.batches = {
        "type": "folded",
        "batch_size": 2,
        "fold_lengths": [150],
        "shape_files": ["${stats_dir}/train/text_shape"],
    }
    lm.dataloader.valid.iter_factory.batches.fold_lengths = [150]
    for split in ("train", "valid"):
        lm.dataloader[split].iter_factory.num_workers = 0
    lm_path = recipe / "lm_training.yaml"
    OmegaConf.save(lm, lm_path)
    prepare_lm_text(lm)
    lm_stages = ["collect_stats", "train"]
    main(
        build_parser(lm_stages).parse_args(
            ["--stages", *lm_stages, "--training_config", str(lm_path)]
        ),
        LMSystem,
        lm_stages,
    )
    assert (recipe / "lm/last.ckpt").is_file()
    run(["infer", "measure"])
    assert (recipe / "experiment/valid.acc.ave_1best.pth").is_file()
    assert (recipe / "inference/test/hyp.scp").is_file()
    scores = json.loads((recipe / "inference/metrics.json").read_text())
    assert scores
