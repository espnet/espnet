# Copyright 2024 Jiatong Shi
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based neural codec task."""

import argparse
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.gan_codec.abs_gan_codec import AbsGANCodec  # noqa
from espnet2.gan_codec.dac.dac import DAC
from espnet2.gan_codec.encodec.encodec import Encodec
from espnet2.gan_codec.espnet_model import ESPnetGANCodecModel
from espnet2.gan_codec.funcodec.funcodec import FunCodec
from espnet2.gan_codec.soundstream.soundstream import SoundStream
from espnet2.tasks.abs_task import AbsTask, optim_classes
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.gan_trainer import GANTrainer
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none  # noqa

codec_choices = ClassChoices(
    "codec",
    classes=dict(
        soundstream=SoundStream,
        encodec=Encodec,
        dac=DAC,
        funcodec=FunCodec,
    ),
    default="soundstream",
)


class GANCodecTask(AbsTask):
    """GAN-based neural codec task."""

    # GAN requires two optimizers
    num_optimizers: int = 2

    # Add variable objects configurations
    class_choices_list = [
        # --codec and --codec_conf
        codec_choices,
    ]

    # Use GANTrainer instead of Trainer
    trainer = GANTrainer

    @classmethod
    @typechecked
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        # NOTE(kamo): Use '_' instead of '-' to avoid confusion
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")  # noqa

        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetGANCodecModel),
            help="The keyword arguments for model class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    @typechecked
    def build_collate_fn(cls, args: argparse.Namespace, train: bool) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        return CommonCollateFn(
            float_pad_value=0.0,
            int_pad_value=0,
        )

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        if args.use_preprocessor:
            # additional check for chunk iterator, to use short utterance in training
            if args.iterator_type == "chunk":
                min_sample_size = args.chunk_length
            else:
                min_sample_size = -1

            retval = CommonPreprocessor(
                train=train,
                token_type=None,  # disable the text process
                speech_name="audio",
                min_sample_size=min_sample_size,
                audio_pad_value=0.0,
                force_single_channel=True,  # NOTE(jiatong): single channel only now
            )
        else:
            retval = None
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("audio",)
        else:
            # Inference mode
            retval = ("audio",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        return ()

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetGANCodecModel:

        # 1. Codec
        codec_class = codec_choices.get_class(args.codec)
        codec = codec_class(**args.codec_conf)

        # 2. Build model
        model = ESPnetGANCodecModel(
            codec=codec,
            **args.model_conf,
        )
        return model

    @classmethod
    def build_optimizers(
        cls,
        args: argparse.Namespace,
        model: ESPnetGANCodecModel,
    ) -> List[torch.optim.Optimizer]:
        # check
        assert hasattr(model.codec, "generator")
        assert hasattr(model.codec, "discriminator")

        # define generator optimizer
        optim_g_class = optim_classes.get(args.optim)
        if optim_g_class is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim}")
        if args.sharded_ddp:
            try:
                import fairscale
            except ImportError:
                raise RuntimeError("Requiring fairscale. Do 'pip install fairscale'")
            optim_g = fairscale.optim.oss.OSS(
                params=model.codec.generator.parameters(),
                optim=optim_g_class,
                **args.optim_conf,
            )
        else:
            optim_g = optim_g_class(
                model.codec.generator.parameters(),
                **args.optim_conf,
            )
        optimizers = [optim_g]

        # define discriminator optimizer
        optim_d_class = optim_classes.get(args.optim2)
        if optim_d_class is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim2}")
        if args.sharded_ddp:
            try:
                import fairscale
            except ImportError:
                raise RuntimeError("Requiring fairscale. Do 'pip install fairscale'")
            optim_d = fairscale.optim.oss.OSS(
                params=model.codec.discriminator.parameters(),
                optim=optim_d_class,
                **args.optim2_conf,
            )
        else:
            optim_d = optim_d_class(
                model.codec.discriminator.parameters(),
                **args.optim2_conf,
            )
        optimizers += [optim_d]

        return optimizers
