"""Task definition for the offline Sortformer speaker-diarization model.

Unlike the generic EEND :class:`~espnet2.tasks.diar.DiarizationTask`, Sortformer
is a single fixed composite architecture (FastConformer + Transformer +
sigmoid head trained with ATS+PIL loss), so this task builds that one model from
a small set of ``*_conf`` dictionaries rather than swappable component choices.

It plugs into ESPnet3 via ``espnet3.utils.task_utils.get_espnet_model``, which
calls :meth:`get_default_config` and :meth:`build_model`. Wire it into a recipe
by naming its dotted path in the training config::

    task: sortformer.task.SortformerDiarizationTask

The recipe directory must be on ``PYTHONPATH`` so that ``sortformer`` is
importable; the package is otherwise self-contained (no NeMo dependency).
"""

import argparse
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.tasks.abs_task import AbsTask
from espnet2.torch_utils.initialize import initialize
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.trainer import Trainer
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none

from .fastconformer_encoder import FastConformerEncoder
from .model import ESPnetSortformerModel
from .preprocessor import MelSpectrogramPreprocessor
from .sortformer_modules import SortformerModules
from .transformer_encoder import TransformerEncoder


class SortformerDiarizationTask(AbsTask):
    """Offline Sortformer diarization task.

    An :class:`~espnet2.tasks.abs_task.AbsTask` subclass that builds the fixed
    Sortformer model (FastConformer encoder + Transformer encoder + Sortformer
    head) from configuration and supplies the training collate function.
    Sortformer is a single fixed composite architecture, so there are no
    swappable component choices; the model is built directly from ``*_conf``.

    The task is selected by dotted path in a recipe's training config::

        task: sortformer.task.SortformerDiarizationTask

    and consumes the following task arguments (each a flat key or a nested
    ``*_conf`` mapping in that config):

    * ``num_spk``: number of speakers (sigmoid head width).
    * ``encoder_conf``: FastConformer encoder kwargs.
    * ``transformer_conf``: Transformer encoder kwargs.
    * ``sortformer_conf``: Sortformer head/projection kwargs.
    * ``preprocessor_conf``: mel-spectrogram preprocessor kwargs.
    * ``model_conf``: model kwargs (e.g. ``ats_weight``, ``pil_weight``).
    * ``init_nest``: optional path to dumped NEST encoder weights used to
      initialize the FastConformer encoder.

    Example config block::

        task: sortformer.task.SortformerDiarizationTask
        num_spk: 4
        init_nest: exp/nest_large_encoder.pt
        encoder_conf:
          d_model: 512
        transformer_conf:
          hidden_size: 192
        sortformer_conf: {}
        preprocessor_conf:
          features: 80
        model_conf:
          ats_weight: 0.5
          pil_weight: 0.5

    The model is then constructed automatically when the training stage is run
    (``python run.py --stages train --training_config conf/training.yaml``).
    """

    num_optimizers: int = 1
    class_choices_list = []
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        """Register the Sortformer task arguments on ``parser``.

        Adds the ``--num_spk`` width plus the nested ``--encoder_conf``,
        ``--transformer_conf``, ``--sortformer_conf``, ``--preprocessor_conf``
        and ``--model_conf`` dictionaries, as well as ``--init`` /
        ``--init_nest`` weight-initialization options. Called by the ESPnet
        argument-parsing machinery; users set these via the training config
        rather than on the command line.

        Args:
            parser: Argument parser to extend in place.
        """
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
        """Build the collate function that batches and pads samples.

        Args:
            args: Parsed task arguments (unused; kept for the AbsTask signature).
            train: Whether the loader is for training (unused here).

        Returns:
            A :class:`~espnet2.train.collate_fn.CommonCollateFn` that pads both
            the float ``speech`` and the 0/1 ``spk_labels`` with ``0.0``.
        """
        # spk_labels are float 0/1 -> pad with 0.0; speech -> pad with 0.0.
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        """Return the per-sample preprocessing function.

        Sortformer extracts features inside the model and the dataset already
        emits ready-to-use arrays (``speech`` plus frame-level ``spk_labels``),
        so no preprocessing is applied.

        Returns:
            ``None`` (no preprocessing).
        """
        # The dataset emits ready-to-use arrays (speech + frame-level labels).
        return None

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """List the data keys each sample must provide.

        Returns ``("speech", "spk_labels")`` for training/validation and
        ``("speech",)`` at inference time, when labels are not available.
        """
        if not inference:
            return ("speech", "spk_labels")
        return ("speech",)

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """List optional data keys (none for this task)."""
        return ()

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetSortformerModel:
        """Construct the Sortformer model from the parsed task arguments.

        Builds the mel preprocessor, FastConformer encoder, Sortformer head and
        Transformer encoder from the corresponding ``*_conf`` mappings, deriving
        shared dimensions (encoder ``d_model``, transformer ``hidden_size``,
        ``n_mels``) from those configs, then assembles them into an
        :class:`~.model.ESPnetSortformerModel`.

        If ``args.init`` is set, standard weight initialization is applied. If
        ``args.init_nest`` points to a dumped NEST encoder state-dict, the
        FastConformer encoder is initialized from those self-supervised weights.

        Args:
            args: Parsed task arguments (see the class docstring for the keys).

        Returns:
            The assembled, optionally initialized Sortformer model.
        """
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
            from .convert_nest import load_nest_encoder

            load_nest_encoder(model, init_nest)
        return model
