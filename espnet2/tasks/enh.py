import argparse
import copy
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from espnet2.diar.layers.abs_mask import AbsMask
from espnet2.diar.layers.multi_mask import MultiMask
from espnet2.diar.separator.tcn_separator_nomask import TCNSeparatorNomask
from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.decoder.conv_decoder import ConvDecoder
from espnet2.enh.decoder.null_decoder import NullDecoder
from espnet2.enh.decoder.stft_decoder import STFTDecoder
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.encoder.conv_encoder import ConvEncoder
from espnet2.enh.encoder.null_encoder import NullEncoder
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.criterions.tf_domain import (
    FrequencyDomainAbsCoherence,
    FrequencyDomainDPCL,
    FrequencyDomainL1,
    FrequencyDomainMSE,
)
from espnet2.enh.loss.criterions.time_domain import (
    CISDRLoss,
    MultiResL1SpecLoss,
    SDRLoss,
    SISNRLoss,
    SNRLoss,
    TimeDomainL1,
    TimeDomainMSE,
)
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.enh.loss.wrappers.dpcl_solver import DPCLSolver
from espnet2.enh.loss.wrappers.fixed_order import FixedOrderSolver
from espnet2.enh.loss.wrappers.multilayer_pit_solver import MultiLayerPITSolver
from espnet2.enh.loss.wrappers.pit_solver import PITSolver
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.enh.separator.asteroid_models import AsteroidModel_Converter
from espnet2.enh.separator.conformer_separator import ConformerSeparator
from espnet2.enh.separator.dan_separator import DANSeparator
from espnet2.enh.separator.dc_crn_separator import DC_CRNSeparator
from espnet2.enh.separator.dccrn_separator import DCCRNSeparator
from espnet2.enh.separator.dpcl_e2e_separator import DPCLE2ESeparator
from espnet2.enh.separator.dpcl_separator import DPCLSeparator
from espnet2.enh.separator.dprnn_separator import DPRNNSeparator
from espnet2.enh.separator.dptnet_separator import DPTNetSeparator
from espnet2.enh.separator.fasnet_separator import FaSNetSeparator
from espnet2.enh.separator.ineube_separator import iNeuBe
from espnet2.enh.separator.tfgridnet_separator import TFGridNet
from espnet2.enh.separator.neural_beamformer import NeuralBeamformer
from espnet2.enh.separator.rnn_separator import RNNSeparator
from espnet2.enh.separator.skim_separator import SkiMSeparator
from espnet2.enh.separator.svoice_separator import SVoiceSeparator
from espnet2.enh.separator.tcn_separator import TCNSeparator
from espnet2.enh.separator.transformer_separator import TransformerSeparator
from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.tasks.abs_task import AbsTask
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.preprocessor import DynamicMixingPreprocessor, EnhPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool, str_or_none

encoder_choices = ClassChoices(
    name="encoder",
    classes=dict(stft=STFTEncoder, conv=ConvEncoder, same=NullEncoder),
    type_check=AbsEncoder,
    default="stft",
)

separator_choices = ClassChoices(
    name="separator",
    classes=dict(
        asteroid=AsteroidModel_Converter,
        conformer=ConformerSeparator,
        dan=DANSeparator,
        dc_crn=DC_CRNSeparator,
        dccrn=DCCRNSeparator,
        dpcl=DPCLSeparator,
        dpcl_e2e=DPCLE2ESeparator,
        dprnn=DPRNNSeparator,
        dptnet=DPTNetSeparator,
        fasnet=FaSNetSeparator,
        rnn=RNNSeparator,
        skim=SkiMSeparator,
        svoice=SVoiceSeparator,
        tcn=TCNSeparator,
        transformer=TransformerSeparator,
        wpe_beamformer=NeuralBeamformer,
        tcn_nomask=TCNSeparatorNomask,
        ineube=iNeuBe,
        tfgridnet=TFGridNet,
    ),
    type_check=AbsSeparator,
    default="rnn",
)

mask_module_choices = ClassChoices(
    name="mask_module",
    classes=dict(multi_mask=MultiMask),
    type_check=AbsMask,
    default="multi_mask",
)

decoder_choices = ClassChoices(
    name="decoder",
    classes=dict(stft=STFTDecoder, conv=ConvDecoder, same=NullDecoder),
    type_check=AbsDecoder,
    default="stft",
)

loss_wrapper_choices = ClassChoices(
    name="loss_wrappers",
    classes=dict(
        pit=PITSolver,
        fixed_order=FixedOrderSolver,
        multilayer_pit=MultiLayerPITSolver,
        dpcl=DPCLSolver,
    ),
    type_check=AbsLossWrapper,
    default=None,
)

criterion_choices = ClassChoices(
    name="criterions",
    classes=dict(
        ci_sdr=CISDRLoss,
        coh=FrequencyDomainAbsCoherence,
        sdr=SDRLoss,
        si_snr=SISNRLoss,
        snr=SNRLoss,
        l1=FrequencyDomainL1,
        dpcl=FrequencyDomainDPCL,
        l1_fd=FrequencyDomainL1,
        l1_td=TimeDomainL1,
        mse=FrequencyDomainMSE,
        mse_fd=FrequencyDomainMSE,
        mse_td=TimeDomainMSE,
        mr_l1_tfd=MultiResL1SpecLoss,
    ),
    type_check=AbsEnhLoss,
    default=None,
)

MAX_REFERENCE_NUM = 100


class EnhancementTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    class_choices_list = [
        # --encoder and --encoder_conf
        encoder_choices,
        # --separator and --separator_conf
        separator_choices,
        # --decoder and --decoder_conf
        decoder_choices,
        # --mask_module and --mask_module_conf
        mask_module_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        # required = parser.get_default("required")

        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetEnhancementModel),
            help="The keyword arguments for model class.",
        )

        group.add_argument(
            "--criterions",
            action=NestedDictAction,
            default=[
                {
                    "name": "si_snr",
                    "conf": {},
                    "wrapper": "fixed_order",
                    "wrapper_conf": {},
                },
            ],
            help="The criterions binded with the loss wrappers.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=False,
            help="Whether to apply preprocessing to data or not",
        )
        group.add_argument(
            "--speech_volume_normalize",
            type=str_or_none,
            default=None,
            help="Scale the maximum amplitude to the given value or range. "
            "e.g. --speech_volume_normalize 1.0 scales it to 1.0.\n"
            "--speech_volume_normalize 0.5_1.0 scales it to a random number in "
            "the range [0.5, 1.0)",
        )
        group.add_argument(
            "--rir_scp",
            type=str_or_none,
            default=None,
            help="The file path of rir scp file.",
        )
        group.add_argument(
            "--rir_apply_prob",
            type=float,
            default=1.0,
            help="THe probability for applying RIR convolution.",
        )
        group.add_argument(
            "--noise_scp",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        group.add_argument(
            "--noise_apply_prob",
            type=float,
            default=1.0,
            help="The probability applying Noise adding.",
        )
        group.add_argument(
            "--noise_db_range",
            type=str,
            default="13_15",
            help="The range of signal-to-noise ratio (SNR) level in decibel.",
        )
        group.add_argument(
            "--short_noise_thres",
            type=float,
            default=0.5,
            help="If len(noise) / len(speech) is smaller than this threshold during "
            "dynamic mixing, a warning will be displayed.",
        )
        group.add_argument(
            "--use_reverberant_ref",
            type=str2bool,
            default=False,
            help="Whether to use reverberant speech references "
            "instead of anechoic ones",
        )
        group.add_argument(
            "--num_spk",
            type=int,
            default=1,
            help="Number of speakers in the input signal.",
        )
        group.add_argument(
            "--num_noise_type",
            type=int,
            default=1,
            help="Number of noise types.",
        )
        group.add_argument(
            "--sample_rate",
            type=int,
            default=8000,
            help="Sampling rate of the data (in Hz).",
        )
        group.add_argument(
            "--force_single_channel",
            type=str2bool,
            default=False,
            help="Whether to force all data to be single-channel.",
        )

        group.add_argument(
            "--dynamic_mixing",
            type=str2bool,
            default=False,
            help="Apply dynamic mixing",
        )

        group.add_argument(
            "--utt2spk",
            type=str_or_none,
            default=None,
            help="The file path of utt2spk file. Only used in dynamic_mixing mode.",
        )

        group.add_argument(
            "--dynamic_mixing_gain_db",
            type=float,
            default=0.0,
            help="Random gain (in dB) for dynamic mixing sources",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()

        return CommonCollateFn(float_pad_value=0.0, int_pad_value=0)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()

        dynamic_mixing = getattr(args, "dynamic_mixing", False)
        use_preprocessor = getattr(args, "use_preprocessor", False)

        assert (
            dynamic_mixing and use_preprocessor
        ) is not True, (
            "'dynamic_mixing' and 'use_preprocessor' should not both be 'True'"
        )

        if dynamic_mixing and train:
            retval = DynamicMixingPreprocessor(
                train=train,
                source_scp=args.train_data_path_and_name_and_type[0][0],
                num_spk=args.separator_conf["num_spk"],
                dynamic_mixing_gain_db=getattr(args, "dynamic_mixing_gain_db", 0.0),
                utt2spk=getattr(args, "utt2spk", None),
            )
        elif use_preprocessor:
            retval = EnhPreprocessor(
                train=train,
                # NOTE(kamo): Check attribute existence for backward compatibility
                rir_scp=args.rir_scp if hasattr(args, "rir_scp") else None,
                rir_apply_prob=args.rir_apply_prob
                if hasattr(args, "rir_apply_prob")
                else 1.0,
                noise_scp=args.noise_scp if hasattr(args, "noise_scp") else None,
                noise_apply_prob=args.noise_apply_prob
                if hasattr(args, "noise_apply_prob")
                else 1.0,
                noise_db_range=args.noise_db_range
                if hasattr(args, "noise_db_range")
                else "13_15",
                short_noise_thres=args.short_noise_thres
                if hasattr(args, "short_noise_thres")
                else 0.5,
                speech_volume_normalize=args.speech_volume_normalize
                if hasattr(args, "speech_volume_normalize")
                else None,
                use_reverberant_ref=args.use_reverberant_ref
                if hasattr(args, "use_reverberant_ref")
                else None,
                num_spk=args.num_spk if hasattr(args, "num_spk") else 1,
                num_noise_type=args.num_noise_type
                if hasattr(args, "num_noise_type")
                else 1,
                sample_rate=args.sample_rate if hasattr(args, "sample_rate") else 8000,
                force_single_channel=args.force_single_channel
                if hasattr(args, "force_single_channel")
                else False,
            )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech_ref1",)
        else:
            # Recognition mode
            retval = ("speech_mix",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ["speech_mix"]
        retval += ["dereverb_ref{}".format(n) for n in range(1, MAX_REFERENCE_NUM + 1)]
        retval += ["speech_ref{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)]
        retval += ["noise_ref{}".format(n) for n in range(1, MAX_REFERENCE_NUM + 1)]
        retval = tuple(retval)
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetEnhancementModel:
        assert check_argument_types()

        encoder = encoder_choices.get_class(args.encoder)(**args.encoder_conf)
        separator = separator_choices.get_class(args.separator)(
            encoder.output_dim, **args.separator_conf
        )
        decoder = decoder_choices.get_class(args.decoder)(**args.decoder_conf)
        if args.separator.endswith("nomask"):
            mask_module = mask_module_choices.get_class(args.mask_module)(
                input_dim=encoder.output_dim,
                **args.mask_module_conf,
            )
        else:
            mask_module = None

        loss_wrappers = []

        if getattr(args, "criterions", None) is not None:
            # This check is for the compatibility when load models
            # that packed by older version
            for ctr in args.criterions:
                criterion_conf = ctr.get("conf", {})
                criterion = criterion_choices.get_class(ctr["name"])(**criterion_conf)
                loss_wrapper = loss_wrapper_choices.get_class(ctr["wrapper"])(
                    criterion=criterion, **ctr["wrapper_conf"]
                )
                loss_wrappers.append(loss_wrapper)

        # 1. Build model
        model = ESPnetEnhancementModel(
            encoder=encoder,
            separator=separator,
            decoder=decoder,
            loss_wrappers=loss_wrappers,
            mask_module=mask_module,
            **args.model_conf,
        )

        # FIXME(kamo): Should be done in model?
        # 2. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model

    @classmethod
    def build_iter_factory(
        cls,
        args: argparse.Namespace,
        distributed_option: DistributedOption,
        mode: str,
        kwargs: dict = None,
    ) -> AbsIterFactory:

        dynamic_mixing = getattr(args, "dynamic_mixing", False)
        if dynamic_mixing and mode == "train":
            args = copy.deepcopy(args)
            args.fold_length = args.fold_length[0:1]

        return super().build_iter_factory(args, distributed_option, mode, kwargs)
