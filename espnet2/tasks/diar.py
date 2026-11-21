import argparse
from typing import Callable, Collection, Dict, List, Optional, Tuple

import humanfriendly
import numpy as np
import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.tcn_encoder import TCNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.diar.attractor.abs_attractor import AbsAttractor
from espnet2.diar.attractor.rnn_attractor import RnnAttractor
from espnet2.diar.decoder.abs_decoder import AbsDecoder
from espnet2.diar.decoder.linear_decoder import LinearDecoder
from espnet2.diar.espnet_model import ESPnetDiarizationModel
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.batch_wav_augment import BatchWavAugment
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.label_aggregation import LabelAggregate
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
    ),
    type_check=AbsFrontend,
    default="default",
    optional=True,
)
wav_augment_choices = ClassChoices(
    name="wav_augment",
    classes=dict(batch_wav=BatchWavAugment),
    type_check=torch.nn.Module,
    default=None,
    optional=True,
)
specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(specaug=SpecAug),
    type_check=AbsSpecAug,
    default=None,
    optional=True,
)
normalize_choices = ClassChoices(
    "normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default="utterance_mvn",
    optional=True,
)
label_aggregator_choices = ClassChoices(
    "label_aggregator",
    classes=dict(label_aggregator=LabelAggregate),
    default="label_aggregator",
)
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        rnn=RNNEncoder,
        tcn=TCNEncoder,
    ),
    type_check=AbsEncoder,
    default="transformer",
)
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(linear=LinearDecoder),
    type_check=AbsDecoder,
    default="linear",
)
attractor_choices = ClassChoices(
    "attractor",
    classes=dict(
        rnn=RnnAttractor,
    ),
    type_check=AbsAttractor,
    default=None,
    optional=True,
)


class DiarizationTask(AbsTask):
    # If you need more than one optimizer, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --wav_augment and --wav_augment_conf
        wav_augment_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # --decoder and --decoder_conf
        decoder_choices,
        # --label_aggregator and --label_aggregator_conf
        label_aggregator_choices,
        # --attractor and --attractor_conf
        attractor_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group("Task related")

        group.add_argument(
            "--num_spk",
            type=int_or_none,
            default=None,
            help="The number fo speakers (for each recording) used in system training",
        )

        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            # wav_augment is the module built from --wav_augment, not a model_conf
            default={
                k: v
                for k, v in get_default_kwargs(ESPnetDiarizationModel).items()
                if k != "wav_augment"
            },
            help="The keyword arguments for model class.",
        )

        group = parser.add_argument_group("Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        # --rir_scp, --rir_apply_prob, --noise_scp, --noise_apply_prob,
        # --noise_db_range and --short_noise_thres, as the other tasks have them
        cls.add_augmentation_arguments(parser)
        group.add_argument(
            "--rir_remove_delay",
            type=str2bool,
            default=True,
            help="Shift the reverberant speech so that the peak of the RIR (the "
            "largest absolute value over its channels) is at time 0, keeping it "
            "aligned with the speaker labels. The peak, not an estimate of the "
            "direct path: with a reflection stronger than the direct sound, the "
            "shift follows the reflection. A multi-channel RIR gets one shift for "
            "all channels (see --rir_delay_per_channel).",
        )
        group.add_argument(
            "--rir_delay_per_channel",
            type=str2bool,
            default=False,
            help="With --rir_remove_delay and a multi-channel RIR, shift each "
            "channel by its own peak: every channel aligned with the labels, the "
            "time differences between the channels lost. False: one shift (the "
            "overall peak) for all channels, which keeps those differences.",
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
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        if args.use_preprocessor:
            # NOTE: getattr keeps configs saved before these options existed loadable
            retval = CommonPreprocessor(
                train=train,
                # RIRs and noises at another rate are resampled to the speech's
                fs=cls._speech_fs(args),
                rir_scp=getattr(args, "rir_scp", None),
                rir_apply_prob=getattr(args, "rir_apply_prob", 1.0),
                rir_remove_delay=getattr(args, "rir_remove_delay", True),
                rir_delay_per_channel=getattr(args, "rir_delay_per_channel", False),
                noise_scp=getattr(args, "noise_scp", None),
                noise_apply_prob=getattr(args, "noise_apply_prob", 1.0),
                noise_db_range=getattr(args, "noise_db_range", "13_15"),
                short_noise_thres=getattr(args, "short_noise_thres", 0.5),
            )
        else:
            retval = None
        return retval

    @staticmethod
    def _speech_fs(args: argparse.Namespace) -> int:
        """Return the sampling rate of the speech in Hz, 0 if unknown.

        frontend_conf fs if given, else the default fs of the frontend class
        (16000 for the default frontend); 0 for a frontend without one.
        """
        fs = (getattr(args, "frontend_conf", None) or {}).get("fs")
        if fs is None and getattr(args, "frontend", None) is not None:
            frontend_class = frontend_choices.get_class(args.frontend)
            fs = get_default_kwargs(frontend_class).get("fs")
        return humanfriendly.parse_size(str(fs)) if fs else 0

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech", "spk_labels")
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        # (Note: jiatong): no optional data names for now
        retval = ()
        return retval

    @staticmethod
    def _check_wav_augment_context(
        args: argparse.Namespace, context: int, fs: Optional[int] = None
    ):
        """Check that the chunks give the speech the context wav_augment removes.

        With context > 0 the model removes the first ``context`` samples of the
        speech (at ``fs``, the rate of wav_augment and the frontend), in
        training and validation. The chunk iterator gives them as
        --chunk_left_context on the speech only, so the labels keep the chunk.
        --chunk_left_context is in samples at --chunk_default_fs when that is
        given, and the iterator scales it to the data's rate; so it must scale
        to exactly ``context`` at ``fs``. Any other data path would leave the
        speech a different number of samples longer than its labels.
        """
        if context <= 0 or getattr(args, "collect_stats", False):
            return
        problems = []
        for mode, it in (
            ("train", args.iterator_type),
            ("valid", getattr(args, "valid_iterator_type", None) or args.iterator_type),
        ):
            if it != "chunk":
                problems.append(f"the {mode} iterator is {it!r}, not 'chunk'")
        left = getattr(args, "chunk_left_context", 0)
        default_fs = getattr(args, "chunk_default_fs", None)
        if default_fs is not None and fs is not None:
            # the iterator gives left * fs // default_fs samples at the data's fs
            given = left * fs // default_fs
            exact = left * fs % default_fs == 0
            where = f" ({left} at --chunk_default_fs {default_fs} = {given} at {fs})"
        else:
            given, exact, where = left, True, ""
        if given != context or not exact:
            problems.append(
                f"--chunk_left_context {left} gives {given} samples{where}, "
                f"not the wav_augment context {context}"
            )
        if "speech" not in getattr(args, "chunk_left_context_keys", ["speech"]):
            problems.append("--chunk_left_context_keys does not include speech")
        if problems:
            raise ValueError(
                "wav_augment_conf context > 0 needs --iterator_type chunk with "
                "--chunk_left_context equal to the context, so the speech chunks "
                "carry the context and the labels do not: " + "; ".join(problems)
            )

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetDiarizationModel:

        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        elif args.input_size is not None and args.frontend is not None:
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = args.input_size + frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        # 2a. Data augmentation for the waveform, in the model (training only)
        if getattr(args, "wav_augment", None) is not None:
            wav_augment_class = wav_augment_choices.get_class(args.wav_augment)
            wav_augment_conf = dict(args.wav_augment_conf)
            # the noise is mixed at the rate of the speech: the frontend's fs
            frontend_fs = (args.frontend_conf or {}).get("fs")
            if frontend_fs is not None:
                frontend_fs = humanfriendly.parse_size(str(frontend_fs))
                if "fs" not in wav_augment_conf:
                    wav_augment_conf["fs"] = frontend_fs
                elif humanfriendly.parse_size(str(wav_augment_conf["fs"])) != (
                    frontend_fs
                ):
                    raise ValueError(
                        f"wav_augment_conf fs={wav_augment_conf['fs']} differs from "
                        f"frontend_conf fs={args.frontend_conf['fs']}"
                    )
            if "fs" in wav_augment_conf:
                wav_augment_conf["fs"] = humanfriendly.parse_size(
                    str(wav_augment_conf["fs"])
                )
            wav_augment = wav_augment_class(**wav_augment_conf)
            cls._check_wav_augment_context(
                args,
                getattr(wav_augment, "context", 0),
                getattr(wav_augment, "fs", None),
            )
        else:
            wav_augment = None

        # 2b. Data augmentation for spectrogram
        if args.specaug is not None:
            specaug_class = specaug_choices.get_class(args.specaug)
            specaug = specaug_class(**args.specaug_conf)
        else:
            specaug = None

        # 3. Normalization layer
        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        # 4. Label Aggregator layer
        label_aggregator_class = label_aggregator_choices.get_class(
            args.label_aggregator
        )
        label_aggregator = label_aggregator_class(**args.label_aggregator_conf)

        # 5. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        # Note(jiatong): Diarization may not use subsampling when processing
        encoder = encoder_class(input_size=input_size, **args.encoder_conf)

        # 6a. Decoder
        decoder_class = decoder_choices.get_class(args.decoder)
        decoder = decoder_class(
            num_spk=args.num_spk,
            encoder_output_size=encoder.output_size(),
            **args.decoder_conf,
        )

        # 6b. Attractor
        if getattr(args, "attractor", None) is not None:
            attractor_class = attractor_choices.get_class(args.attractor)
            attractor = attractor_class(
                encoder_output_size=encoder.output_size(),
                **args.attractor_conf,
            )
        else:
            attractor = None

        # 7. Build model
        model = ESPnetDiarizationModel(
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            label_aggregator=label_aggregator,
            encoder=encoder,
            decoder=decoder,
            attractor=attractor,
            wav_augment=wav_augment,
            **{k: v for k, v in args.model_conf.items() if k != "wav_augment"},
        )

        # FIXME(kamo): Should be done in model?
        # 8. Initialize
        if args.init is not None:
            initialize(model, args.init)

        return model
