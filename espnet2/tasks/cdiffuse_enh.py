import argparse
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.enh.cdiffuse_espnet_model import ESPnetEnhancementModel

from espnet2.tasks.abs_task import AbsTask
from espnet2.tasks.enh import encoder_choices
from espnet2.tasks.enh import separator_choices
from espnet2.tasks.enh import decoder_choices
from espnet2.tasks.enh import loss_wrapper_choices
from espnet2.tasks.enh import criterion_choices
from espnet2.tasks.enh import EnhancementTask as Orig_EnhancementTask
from espnet2.torch_utils.initialize import initialize


class EnhancementTask(Orig_EnhancementTask, AbsTask):
    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetEnhancementModel:
        assert check_argument_types()

        encoder = encoder_choices.get_class(args.encoder)(**args.encoder_conf)
        separator = separator_choices.get_class(args.separator)(
            encoder.output_dim, **args.separator_conf
        )
        decoder = decoder_choices.get_class(args.decoder)(**args.decoder_conf)

        loss_wrappers = []
        for ctr in args.criterions:
            criterion = criterion_choices.get_class(ctr["name"])(**ctr["conf"])
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
            **args.model_conf
        )

        # FIXME(kamo): Should be done in model?
        # 2. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model
