"""Exercise the stock ASR and LM stages using a small CPU corpus."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from espnet3.utils.config_utils import load_and_merge_config

RECIPE = Path(__file__).resolve().parents[4] / "egs3/voices/esp2_asr"


@pytest.mark.execution_timeout(180)
def test_complete_stage_pipeline_on_fixture(corpus, monkeypatch):
    """Train, load and score ASR with a nonzero trained LM fusion weight."""
    import json
    import shutil

    import yaml
    from hydra.utils import instantiate

    from egs3.voices.esp2_asr.src import tokenizer as tokenizer_module
    from egs3.voices.esp2_asr.src.language_model import prepare_lm_text
    from espnet3.publication import InferenceModel
    from espnet3.publication.demo.assets import DEFAULT_UI_ASSETS
    from espnet3.publication.demo.session import DemoSession
    from espnet3.systems.esp2_asr.system import ASRSystem
    from espnet3.systems.esp2_lm.system import ESP2LMSystem
    from espnet3.systems.launch import launch
    from espnet3.utils.publication_utils import pack_model

    recipe, source = corpus
    monkeypatch.chdir(recipe)
    monkeypatch.setattr(
        tokenizer_module,
        "gather_training_text",
        lambda **_: [" ".join("ABCDEFGHIJKLMNOPQRSTUVWXYZ")] * 5,
    )
    training = load_and_merge_config(
        RECIPE / "conf/training_devkit.yaml", "training.yaml", resolve=False
    )
    training.recipe_dir = "."
    training.data_dir = str(Path(".") / "data")
    training.exp_dir = str(Path(".") / "experiment")
    training.stats_dir = str(Path(".") / "statistics")
    training.create_dataset = {"recipe_dir": str(recipe), "source_dir": str(source)}
    for split in ("train", "valid"):
        training.dataset[split][0].data_src = "egs3.voices.esp2_asr.dataset"
        training.dataset[split][0].data_src_args.recipe_dir = str(recipe)
        training.dataloader[split].iter_factory.num_workers = 0
    training.tokenizer.vocab_size = 30
    training.tokenizer.save_path = str(Path(".") / "tokenizer")
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
        accumulate_grad_batches=1,
    )
    inference = load_and_merge_config(
        RECIPE / "conf/inference.yaml", "inference.yaml", resolve=False
    )
    inference.recipe_dir = "."
    inference.exp_dir = str(Path(".") / "experiment")
    inference.inference_dir = str(Path(".") / "inference")
    inference.batch_size = 1
    inference.model.update(
        asr_model_file=str(Path(".") / "experiment/valid.acc.ave_1best.pth"),
        beam_size=2,
        maxlenratio=0.1,
        lm_train_config=str(Path(".") / "lm/config.yaml"),
        lm_file=str(Path(".") / "lm/valid.loss.ave_1best.pth"),
    )
    for entry in inference.dataset.test:
        entry.data_src = "egs3.voices.esp2_asr.dataset"
        entry.data_src_args.recipe_dir = str(recipe)
        entry.data_src_args.limit = 1
    metrics = load_and_merge_config(
        RECIPE / "conf/metrics.yaml", "metrics.yaml", resolve=False
    )
    metrics.metrics[2].metric.bpemodel = str(Path(".") / "tokenizer/unigram.model")
    configs = {"training": training, "inference": inference, "metrics": metrics}
    for name, config in configs.items():
        OmegaConf.save(config, recipe / f"{name}.yaml")

    def run(stages):
        argv = ["--stages", *stages]
        for name in configs:
            argv += [f"--{name}_config", str(Path(".") / f"{name}.yaml")]
        launch(ASRSystem, argv=argv)

    # Reload YAML for train: stock collect_stats removes normalization in memory.
    run(["create_dataset", "train_tokenizer", "collect_stats"])
    assert (recipe / "statistics/train/feats_shape").is_file()
    run(["train"])
    assert (
        yaml.safe_load((recipe / "experiment/config.yaml").read_text())["normalize"]
        == "global_mvn"
    )
    lm = load_and_merge_config(
        RECIPE / "conf/training_lm.yaml",
        "training.yaml",
        default_package="egs3.TEMPLATE.esp2_lm",
        resolve=False,
    )
    lm.recipe_dir = "."
    lm.data_dir = str(Path(".") / "data")
    lm.exp_dir = str(Path(".") / "lm")
    lm.tokenizer_dir = str(Path(".") / "tokenizer")
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
    launch(
        ESP2LMSystem,
        argv=["--stages", *lm_stages, "--training_config", str(lm_path)],
    )
    assert (recipe / "lm/last.ckpt").is_file()
    run(["infer", "measure"])
    assert (recipe / "experiment/valid.acc.ave_1best.pth").is_file()
    assert (recipe / "inference/test/text.scp").is_file()
    scores = json.loads((recipe / "inference/metrics.json").read_text())
    assert scores

    # Each scorer consumes the raw dataset text and the same indexed hypotheses.
    for entry in inference.dataset.test:
        split = entry.name
        output = recipe / "inference" / split
        hypotheses = dict(
            line.split(" ", 1)
            for line in (output / "text.scp").read_text().splitlines()
        )
        references = dict(
            line.split(" ", 1)
            for line in (output / "dataset/text.scp").read_text().splitlines()
        )
        assert list(hypotheses) == list(references)
        for specification in metrics.metrics:
            metric = instantiate(specification.metric)
            ref, hyp = list(references.values()), list(hypotheses.values())
            metric.update(ref=ref, hyp=hyp)
            expected = metric.compute()
            assert expected == scores[specification.metric._target_][split]
            metric.reset()
            for reference, hypothesis in zip(ref, hyp):
                metric.update(ref=[reference], hyp=[hypothesis])
            assert metric.compute() == expected

    # Pack real ASR+LM checkpoints and relocate the bundle before loading it.
    publication = load_and_merge_config(
        RECIPE / "conf/publication.yaml", "publication.yaml", resolve=False
    )
    publication.recipe_dir = "."
    publication.exp_dir = str(Path(".") / "experiment")
    publication.inference_dir = str(Path(".") / "inference")
    publication.tokenizer_dir = str(Path(".") / "tokenizer")
    publication.stats_dir = str(Path(".") / "statistics")
    publication.lm_exp_dir = str(Path(".") / "lm")
    publication.pack_model.files.language_model = str(
        Path(".") / "lm/valid.loss.ave_1best.pth"
    )
    publication.pack_model.out_dir = str(Path(".") / "model_pack")
    publication.pack_model.include = []
    publication.pack_model.readme = None
    training.recipe_dir = "."
    inference.recipe_dir = "."
    bundle = pack_model(training, publication, inference, metrics)
    relocated = recipe / "relocated_pack"
    shutil.move(bundle, relocated)
    from espnet3.systems.base.inference_provider import InferenceProvider

    sample_config = OmegaConf.create(OmegaConf.to_container(inference, resolve=True))
    sample_config.test_set = "test"
    sample = InferenceProvider.build_dataset(sample_config)[0]
    expected = (
        (recipe / "inference/test/text.scp")
        .read_text()
        .splitlines()[0]
        .split(" ", 1)[1]
    )
    for path in (
        recipe / "experiment",
        recipe / "lm",
        recipe / "tokenizer",
        recipe / "statistics",
    ):
        path.rename(path.with_name(path.name + "_unavailable"))
    packed = InferenceModel.from_packed(relocated, device="cpu")
    with pytest.raises(TypeError, match="Unknown inference input"):
        packed(sample, idx=0)
    assert packed({"speech": sample["speech"]}, idx=0) == {
        "text": expected,
        "utt_id": "0",
    }
    demo = DemoSession(
        relocated,
        OmegaConf.create(
            {"model": {"call_args": {}}, "ui": {"title": None, "description": None}}
        ),
        packed,
        DEFAULT_UI_ASSETS.clone(),
    )
    callback = demo.create_inference_fn(demo.input_specs, demo.output_specs)
    assert callback((16000, sample["speech"])) == expected
