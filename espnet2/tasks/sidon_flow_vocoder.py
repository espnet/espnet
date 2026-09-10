"""Task definition for the flow-matching Sidon vocoder (no discriminator)."""

from espnet2.enh.decoder.sidon_vocoder import build_vocoder
from espnet2.enh.sidon_flow_vocoder_model import SidonFlowVocoderModel
from espnet2.enh.sidon_model import build_ssl_encoder
from espnet2.tasks.abs_task import AbsTask
from espnet2.tasks.sidon_vocoder import SidonVocoderTask
from espnet2.train.trainer import Trainer


class SidonFlowVocoderTask(AbsTask):
    """Same data, collate and encoder options as SidonVocoderTask, but one
    optimizer and the plain Trainer. Requires ``--vocoder_type cfm``."""

    num_optimizers = 1
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser):
        SidonVocoderTask.add_task_arguments(parser)
        group = parser.add_argument_group("ESPnet-Sidon flow vocoder")
        group.add_argument(
            "--sigma_min",
            type=float,
            default=1e-4,
            help="residual noise scale at t=1 of the flow-matching path",
        )

    build_collate_fn = SidonVocoderTask.build_collate_fn
    build_preprocess_fn = SidonVocoderTask.build_preprocess_fn
    required_data_names = SidonVocoderTask.required_data_names
    optional_data_names = SidonVocoderTask.optional_data_names

    @classmethod
    def build_model(cls, args):
        if args.vocoder_type != "cfm":
            raise ValueError(
                "SidonFlowVocoderTask trains the flow-matching vocoder: set "
                f"--vocoder_type cfm (got {args.vocoder_type!r}); dac and hifigan "
                "train adversarially with enh_train_sidon_vocoder"
            )
        encoder = build_ssl_encoder(
            args.ssl_encoder,
            args.ssl_encoder_conf,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            input_sr=args.input_sr,
        )
        if args.fp_model_path:
            SidonVocoderTask._load_feature_predictor(encoder, args.fp_model_path)
        elif args.use_predicted_feat:
            raise ValueError("--use_predicted_feat true requires --fp_model_path")
        vocoder = build_vocoder(args.vocoder_type, encoder.ssl_dim, args.vocoder_conf)
        return SidonFlowVocoderModel(
            ssl_encoder=encoder,
            vocoder=vocoder,
            use_predicted_feat=args.use_predicted_feat,
            input_sr=args.input_sr,
            output_sr=args.output_sr,
            segment_duration=args.segment_duration,
            sigma_min=args.sigma_min,
        )
