"""Task definition for GAN-based speech-tokenizer training."""

import argparse
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.tok.espnet_model import ESPnetGANSpeechTokenizerModel
from espnet2.tok.modules.token_embedding import TokenEmbedding
from espnet2.tok.objective.asr import ASRObjective
from espnet2.tok.objective.reconstruction.hifigan import (
    HiFiGANReconstructionObjective,
)
from espnet2.tok.quantizer.abs_quantizer import (
    AbsSpeechTokenizerQuantizer,
)
from espnet2.tok.quantizer.differentiable_kmeans import (
    DifferentiableKMeans,
)
from espnet2.tok.tokenizer import SpeechTokenizer
from espnet2.tasks.abs_task import AbsTask, optim_classes
from espnet2.tasks.asr import (
    ASRTask,
    decoder_choices,
    encoder_choices,
    frontend_choices,
    normalize_choices,
    postencoder_choices,
    preencoder_choices,
    specaug_choices,
)
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.tok_gan_trainer import SpeechTokenizerGANTrainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool, str_or_none

quantizer_choices = ClassChoices(
    "quantizer",
    classes={"differentiable_kmeans": DifferentiableKMeans},
    type_check=AbsSpeechTokenizerQuantizer,
    default="differentiable_kmeans",
)
reconstruction_objective_choices = ClassChoices(
    "reconstruction_objective",
    classes={"hifigan": HiFiGANReconstructionObjective},
    default="hifigan",
)


def _prefixed_choices(name: str, choices: ClassChoices) -> ClassChoices:
    """Copy ASR component choices under a speech-tokenizer-specific name."""
    return ClassChoices(
        name=name,
        classes=choices.classes,
        type_check=choices.base_type,
        default=choices.default,
        optional=choices.optional,
    )


asr_specaug_choices = _prefixed_choices("asr_specaug", specaug_choices)
asr_normalize_choices = _prefixed_choices("asr_normalize", normalize_choices)
asr_preencoder_choices = _prefixed_choices("asr_preencoder", preencoder_choices)
asr_encoder_choices = _prefixed_choices("asr_encoder", encoder_choices)
asr_postencoder_choices = _prefixed_choices("asr_postencoder", postencoder_choices)
asr_decoder_choices = _prefixed_choices("asr_decoder", decoder_choices)


class GANSpeechTokenizerTask(AbsTask):
    """Build adversarial ASR and reconstruction training with two optimizers."""

    num_optimizers: int = 2
    trainer = SpeechTokenizerGANTrainer
    class_choices_list = [
        frontend_choices,
        quantizer_choices,
        asr_specaug_choices,
        asr_normalize_choices,
        asr_preencoder_choices,
        asr_encoder_choices,
        asr_postencoder_choices,
        asr_decoder_choices,
        reconstruction_objective_choices,
    ]

    @classmethod
    @typechecked
    def add_task_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add tokenizer, ASR, reconstruction, and preprocessing arguments."""
        parser.get_default("required")[:] += ["token_list"]
        group = parser.add_argument_group("Task related")
        group.add_argument("--token_list", type=str_or_none, default=None)
        group.add_argument(
            "--tokenizer_conf",
            action=NestedDictAction,
            default=get_default_kwargs(SpeechTokenizer),
        )
        group.add_argument(
            "--asr_embedding_conf",
            action=NestedDictAction,
            default={
                "embed_dim": 512,
                "use_positional_encoding": True,
                "positional_dropout_rate": 0.1,
            },
        )
        group.add_argument(
            "--asr_model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetASRModel),
        )
        group.add_argument(
            "--asr_ctc_conf",
            action=NestedDictAction,
            default=get_default_kwargs(CTC),
        )
        group.add_argument(
            "--asr_joint_net_conf", action=NestedDictAction, default=None
        )
        group.add_argument("--asr_init", type=str_or_none, default=None)
        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetGANSpeechTokenizerModel),
        )

        preprocess = parser.add_argument_group("Preprocess related")
        preprocess.add_argument("--use_preprocessor", type=str2bool, default=True)
        preprocess.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word", "phn"],
        )
        preprocess.add_argument("--bpemodel", type=str_or_none, default=None)
        preprocess.add_argument(
            "--non_linguistic_symbols", type=str_or_none, default=None
        )
        preprocess.add_argument("--cleaner", type=str_or_none, default=None)
        preprocess.add_argument(
            "--g2p", type=str_or_none, choices=g2p_choices, default=None
        )
        for choices in cls.class_choices_list:
            choices.add_arguments(group)

    @classmethod
    @typechecked
    def build_collate_fn(cls, args: argparse.Namespace, train: bool) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        return CommonCollateFn(
            float_pad_value=0.0, int_pad_value=-1, not_sequence=["spembs"]
        )

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        if not args.use_preprocessor:
            return None
        return CommonPreprocessor(
            train=train,
            token_type=args.token_type,
            token_list=args.token_list,
            bpemodel=args.bpemodel,
            non_linguistic_symbols=args.non_linguistic_symbols,
            text_cleaner=args.cleaner,
            g2p_type=args.g2p,
        )

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        return ("speech",) if inference else ("speech", "text", "spembs")

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        return ()

    @staticmethod
    def _load_token_list(token_list) -> List[str]:
        if isinstance(token_list, str):
            with open(token_list, encoding="utf-8") as file:
                return [line.rstrip() for line in file]
        if isinstance(token_list, (tuple, list)):
            return list(token_list)
        raise RuntimeError("token_list must be a path, tuple, or list")

    @classmethod
    def _build_asr_model(
        cls, args: argparse.Namespace, token_list: List[str], input_size: int
    ) -> ESPnetASRModel:
        """Reuse ASRTask construction with an externally embedded input."""
        asr_args = argparse.Namespace(
            token_list=token_list,
            model_conf=dict(args.asr_model_conf),
            input_size=input_size,
            frontend=None,
            frontend_conf={},
            specaug=args.asr_specaug,
            specaug_conf=args.asr_specaug_conf,
            normalize=args.asr_normalize,
            normalize_conf=args.asr_normalize_conf,
            preencoder=args.asr_preencoder,
            preencoder_conf=args.asr_preencoder_conf,
            encoder=args.asr_encoder,
            encoder_conf=args.asr_encoder_conf,
            postencoder=args.asr_postencoder,
            postencoder_conf=args.asr_postencoder_conf,
            decoder=args.asr_decoder,
            decoder_conf=args.asr_decoder_conf,
            ctc_conf=args.asr_ctc_conf,
            joint_net_conf=args.asr_joint_net_conf,
            model="espnet",
            init=args.asr_init,
        )
        return ASRTask.build_model(asr_args)

    @classmethod
    def _build_asr_objective(
        cls,
        args: argparse.Namespace,
        token_list: List[str],
        num_tokens: int,
    ) -> ASRObjective:
        """Build the ASR token embedding and objective."""
        embedding_conf = dict(args.asr_embedding_conf)
        embedding_conf.pop("input_size", None)
        embedding = TokenEmbedding(input_size=num_tokens, **embedding_conf)
        asr_model = cls._build_asr_model(args, token_list, embedding.output_size())
        return ASRObjective(asr_model, embedding)

    @classmethod
    def _build_reconstruction_objective(
        cls,
        args: argparse.Namespace,
        num_tokens: int,
    ) -> HiFiGANReconstructionObjective:
        """Build the configured waveform-reconstruction objective."""
        reconstruction_conf = dict(args.reconstruction_objective_conf)
        reconstruction_conf.pop("num_tokens", None)
        objective_class = reconstruction_objective_choices.get_class(
            args.reconstruction_objective
        )
        return objective_class(num_tokens=num_tokens, **reconstruction_conf)

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetGANSpeechTokenizerModel:
        """Assemble the shared tokenizer and both downstream objectives."""
        token_list = cls._load_token_list(args.token_list)
        args.token_list = token_list

        frontend = frontend_choices.get_class(args.frontend)(**args.frontend_conf)
        quantizer = quantizer_choices.get_class(args.quantizer)(**args.quantizer_conf)
        tokenizer = SpeechTokenizer(frontend, quantizer, **args.tokenizer_conf)

        asr_objective = cls._build_asr_objective(
            args, token_list, quantizer.num_clusters
        )
        reconstruction_objective = cls._build_reconstruction_objective(
            args, quantizer.num_clusters
        )
        return ESPnetGANSpeechTokenizerModel(
            tokenizer=tokenizer,
            asr_objective=asr_objective,
            reconstruction_objective=reconstruction_objective,
            **args.model_conf,
        )

    @classmethod
    def build_optimizers(
        cls, args: argparse.Namespace, model: ESPnetGANSpeechTokenizerModel
    ) -> List[torch.optim.Optimizer]:
        """Separate all main parameters from discriminator-only parameters."""
        discriminator_parameters = list(
            model.reconstruction_objective.discriminator.parameters()
        )
        discriminator_ids = {id(parameter) for parameter in discriminator_parameters}
        main_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in discriminator_ids
        ]
        discriminator_parameters = [
            parameter
            for parameter in discriminator_parameters
            if parameter.requires_grad
        ]
        if not main_parameters or not discriminator_parameters:
            raise RuntimeError("Both main and discriminator parameters are required")

        optim_class = optim_classes.get(args.optim)
        optim2_class = optim_classes.get(args.optim2)
        if optim_class is None or optim2_class is None:
            raise ValueError("Unknown main or discriminator optimizer")
        if args.sharded_ddp:
            try:
                import fairscale
            except ImportError as error:
                raise RuntimeError("Requiring fairscale") from error
            main_optimizer = fairscale.optim.oss.OSS(
                params=main_parameters, optim=optim_class, **args.optim_conf
            )
            discriminator_optimizer = fairscale.optim.oss.OSS(
                params=discriminator_parameters,
                optim=optim2_class,
                **args.optim2_conf,
            )
        else:
            main_optimizer = optim_class(main_parameters, **args.optim_conf)
            discriminator_optimizer = optim2_class(
                discriminator_parameters, **args.optim2_conf
            )
        return [main_optimizer, discriminator_optimizer]
