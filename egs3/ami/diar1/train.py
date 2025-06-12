import argparse

import lightning as L
from hydra.utils import instantiate
from omegaconf import OmegaConf
from espnet3.utils.config import run_stage
from espnet3 import get_espnet_model, save_espnet_config
from espnet3.trainer import ESPnetEZLightningTrainer, LitESPnetModel
from functools import partial


def load_line(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--stage", type=int, default=-1)
    parser.add_argument("--stop-stage", type=int, default=-1)
    parser.add_argument("--skip-stages", type=str, default="-1,")
    args = parser.parse_args()

    skip_stages = [int(x) for x in args.skip_stages.split(",") if len(x)]

    run_stage_flag = partial(
        run_stage,
        start_stage=args.stage,
        stop_stage=args.stop_stage,
        skip_stages=skip_stages,
    )

    # Load config
    #OmegaConf.register_new_resolver("load_line", load_line)
    config = OmegaConf.load(args.config)

    # Set seed
    if getattr(config, "seed", None) is not None:
        assert isinstance(config.seed, int), "seed should be an integer"
        L.seed_everything(config.seed)

    if run_stage_flag(0):
        # run data sim
        from dataset.simulate_multispk_data import (
            create_librispeech_conversation_dataset,
        )
        create_librispeech_conversation_dataset(config.synth_multispk_data)

    if run_stage_flag(1):
        # get manifests for AMI
        from dataset.create_dataset import get_ami_manifets
        get_ami_manifets(config)

    if run_stage_flag(2):
        from dataset.create_dataset import get_cuts
        get_cuts(config)
        # create cuts with desired chunk size

    model = instantiate(config.model)
    lit_model = LitESPnetModel(model, config)

    # Setup trainer
    trainer = ESPnetEZLightningTrainer(
        model=lit_model,
        expdir=config.expdir,
        config=config.trainer,
        best_model_criterion=config.best_model_criterion,
    )

    fit_params = {} if not hasattr(config, "fit") else config.fit
    trainer.fit(**fit_params)


if __name__ == "__main__":
    main()
