from pathlib import Path

from espnet3.utils.config_utils import (
    load_and_merge_config,
    load_default_config,
)


def test_load_default_config_train_contains_expected_targets() -> None:
    cfg = load_default_config("training.yaml", "egs3.TEMPLATE.codec")
    assert (
        cfg.dataset._target_ == "espnet3.components.data.data_organizer.DataOrganizer"
    )
    assert cfg.dataset.recipe_dir == "."
    assert cfg.optimizers.generator.params == "generator"
    assert cfg.optimizers.discriminator.params == "discriminator"
    assert cfg.schedulers.generator.interval == "epoch"
    assert cfg.trainer.gan.generator_first is False
    assert (
        cfg.dataloader.train.iter_factory._target_
        == "espnet2.iterators.chunk_iter_factory.ChunkIterFactory"
    )


def test_load_and_merge_config_user_overrides_template_defaults(
    tmp_path: Path,
) -> None:
    user = tmp_path / "train_user.yaml"
    user.write_text(
        """
exp_tag: user_train
model:
  codec: encodec
  codec_conf: {}
optimizers:
  generator:
    optimizer:
      _target_: torch.optim.SGD
      lr: 0.1
    params: generator
  discriminator:
    optimizer:
      _target_: torch.optim.SGD
      lr: 0.1
    params: discriminator
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = load_and_merge_config(
        user,
        "training.yaml",
        default_package="egs3.TEMPLATE.codec",
    )

    assert cfg is not None
    # user config should override template default values
    assert cfg.optimizers.generator.optimizer._target_ == "torch.optim.SGD"
    assert cfg.optimizers.generator.optimizer.lr == 0.1
    # template default fields not touched by the user config should remain
    assert cfg.parallel.env == "local"
    assert cfg.trainer.gan.generator_first is False
