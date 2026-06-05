"""Task definition for the offline Sortformer speaker-diarization model.

Unlike the generic EEND :class:`~espnet2.tasks.diar.DiarizationTask`, Sortformer
is a single fixed composite architecture (FastConformer + Transformer +
sigmoid head trained with ATS+PIL loss), so this task builds that one model from
a small set of ``*_conf`` dictionaries rather than swappable component choices.

It plugs into ESPnet3 via ``espnet3.utils.task_utils.get_espnet_model``, which
calls :meth:`get_default_config` and :meth:`build_model`.
"""

import argparse
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.diar.espnet_sortformer_model import ESPnetSortformerModel
from espnet2.diar.sortformer.fastconformer_encoder import FastConformerEncoder
from espnet2.diar.sortformer.preprocessor import MelSpectrogramPreprocessor
from espnet2.diar.sortformer.sortformer_modules import SortformerModules
from espnet2.diar.sortformer.transformer_encoder import TransformerEncoder
from espnet2.tasks.abs_task import AbsTask
from espnet2.torch_utils.initialize import initialize
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.trainer import Trainer
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none


class SortformerDiarizationTask(AbsTask):
    """Offline Sortformer diarization task.

    Sortformer is a single fixed composite architecture, so there are no
    swappable component choices; the model is built directly from ``*_conf``.
    """

    num_optimizers: int = 1
    class_choices_list = []
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        group.add_argument(
            "--num_spk",
            type=int,
            default=4,
            help="Number of speakers (sigmoid head width).",
        )
        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="Weight initialization method (None keeps default init).",
        )
        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="Unused (kept for ESPnet config compatibility).",
        )
        group.add_argument(
            "--encoder_conf",
            action=NestedDictAction,
            default=dict(),
            help="FastConformer encoder kwargs.",
        )
        group.add_argument(
            "--transformer_conf",
            action=NestedDictAction,
            default=dict(),
            help="Transformer encoder kwargs.",
        )
        group.add_argument(
            "--sortformer_conf",
            action=NestedDictAction,
            default=dict(),
            help="SortformerModules kwargs (head/projection).",
        )
        group.add_argument(
            "--preprocessor_conf",
            action=NestedDictAction,
            default=dict(),
            help="Mel-spectrogram preprocessor kwargs.",
        )
        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=dict(),
            help="Model kwargs (e.g. ats_weight, pil_weight).",
        )
        group.add_argument(
            "--init_nest",
            type=str_or_none,
            default=None,
            help="Path to a NEST encoder state-dict (.pt) to initialize the "
            "FastConformer encoder from (e.g. nvidia/ssl_en_nest_large_v1.0).",
        )
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=False,
            help="Sortformer extracts features inside the model; no espnet "
            "text/audio preprocessor is needed.",
        )

    @classmethod
    @typechecked
    def build_collate_fn(cls, args: argparse.Namespace, train: bool) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        # spk_labels are float 0/1 -> pad with 0.0; speech -> pad with 0.0.
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        # The dataset emits ready-to-use arrays (speech + frame-level labels).
        return None

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            return ("speech", "spk_labels")
        return ("speech",)

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        return ()

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetSortformerModel:
        num_spk = args.num_spk
        encoder_conf = dict(args.encoder_conf)
        transformer_conf = dict(args.transformer_conf)
        sortformer_conf = dict(args.sortformer_conf)
        preprocessor_conf = dict(args.preprocessor_conf)
        model_conf = dict(args.model_conf)

        # Derive shared dims (FastConformer d_model, Transformer d_model).
        fc_d_model = encoder_conf.get("d_model", 512)
        tf_d_model = transformer_conf.get("hidden_size", 192)
        n_mels = preprocessor_conf.get("features", 80)

        preprocessor = MelSpectrogramPreprocessor(**preprocessor_conf)
        encoder = FastConformerEncoder(feat_in=n_mels, **encoder_conf)
        sortformer_modules = SortformerModules(
            num_spks=num_spk,
            fc_d_model=fc_d_model,
            tf_d_model=tf_d_model,
            **{
                k: v
                for k, v in sortformer_conf.items()
                if k not in ("fc_d_model", "tf_d_model", "num_spks")
            },
        )
        transformer_encoder = TransformerEncoder(**transformer_conf)

        model = ESPnetSortformerModel(
            preprocessor=preprocessor,
            encoder=encoder,
            sortformer_modules=sortformer_modules,
            transformer_encoder=transformer_encoder,
            num_spk=num_spk,
            **model_conf,
        )

        if args.init is not None:
            initialize(model, args.init)

        # Initialize the FastConformer from NEST self-supervised weights.
        init_nest = getattr(args, "init_nest", None)
        if init_nest is not None:
            from espnet2.diar.sortformer.convert_nest import load_nest_encoder

            load_nest_encoder(model, init_nest)
        return model
