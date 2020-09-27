# Copyright 2020 The ESPnet Authors.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Backend module for `asr_train.py`."""

import json
import logging
import os

from chainer import training
from chainer.training import extensions
from tensorboardX import SummaryWriter
import torch

from espnet.asr.asr_utils import adadelta_eps_decay
from espnet.asr.asr_utils import CompareValueTrigger
from espnet.asr.asr_utils import format_mulenc_args
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import restore_snapshot
from espnet.asr.asr_utils import snapshot_object
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_snapshot
from espnet.asr.pytorch_backend.asr import CustomEvaluator
from espnet.asr.pytorch_backend.asr import CustomUpdater
from espnet.asr.pytorch_backend.asr_init import freeze_modules
from espnet.asr.pytorch_backend.asr_init import load_trained_modules
from espnet.asr.pytorch_backend.converter import CustomConverter
from espnet.asr.pytorch_backend.converter import CustomConverterMulEnc
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.lm import default
from espnet.utils.dataset import ChainerDataLoader
from espnet.utils.dataset import TransformDataset
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.tensorboard_logger import TensorboardLogger
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop


def train(args):
    """Train with the given args.

    Args:
        args (namespace): The program arguments.

    """
    set_deterministic_pytorch(args)
    if args.num_encs > 1:
        args = format_mulenc_args(args)

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning("cuda is not available")

    # get input and output dimension info
    with open(args.valid_json, "rb") as f:
        valid_json = json.load(f)["utts"]
    utts = list(valid_json.keys())
    idim_list = [
        int(valid_json[utts[0]]["input"][i]["shape"][-1]) for i in range(args.num_encs)
    ]
    odim = int(valid_json[utts[0]]["output"][0]["shape"][-1])
    if hasattr(args, "decoder_mode") and args.decoder_mode == "maskctc":
        odim += 1  # for the <mask> token
    for i in range(args.num_encs):
        logging.info("stream{}: input dims : {}".format(i + 1, idim_list[i]))
    logging.info("#output dims: " + str(odim))

    # specify attention, CTC, hybrid mode
    if "transducer" in args.model_module:
        if (
            getattr(args, "etype", False) == "transformer"
            or getattr(args, "dtype", False) == "transformer"
        ):
            mtl_mode = "transformer_transducer"
        else:
            mtl_mode = "transducer"
        logging.info("Pure transducer mode")
    elif args.mtlalpha == 1.0:
        mtl_mode = "ctc"
        logging.info("Pure CTC mode")
    elif args.mtlalpha == 0.0:
        mtl_mode = "att"
        logging.info("Pure attention mode")
    else:
        mtl_mode = "mtl"
        logging.info("Multitask learning mode")

    if (args.enc_init is not None or args.dec_init is not None) and args.num_encs == 1:
        model = load_trained_modules(idim_list[0], odim, args)
    else:
        model_class = dynamic_import(args.model_module)
        model = model_class(
            idim_list[0] if args.num_encs == 1 else idim_list, odim, args
        )
    assert isinstance(model, ASRInterface)

    logging.info(
        " Total parameter of the model = "
        + str(sum(p.numel() for p in model.parameters()))
    )

    if args.rnnlm is not None:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        rnnlm = default.ClassifierWithState(
            default.RNNLM(len(args.char_list), rnnlm_args.layer, rnnlm_args.unit)
        )
        torch_load(args.rnnlm, rnnlm)
        model.rnnlm = rnnlm

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + "/model.json"
    with open(model_conf, "wb") as f:
        logging.info("writing a model config file to " + model_conf)
        f.write(
            json.dumps(
                (idim_list[0] if args.num_encs == 1 else idim_list, odim, vars(args)),
                indent=4,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf_8")
        )
    for key in sorted(vars(args).keys()):
        logging.info("ARGS: " + key + ": " + str(vars(args)[key]))

    reporter = model.reporter

    # check the use of multi-gpu
    if args.ngpu > 1:
        if args.batch_size != 0:
            logging.warning(
                "batch size is automatically increased (%d -> %d)"
                % (args.batch_size, args.batch_size * args.ngpu)
            )
            args.batch_size *= args.ngpu
        if args.num_encs > 1:
            # TODO(ruizhili): implement data parallel for multi-encoder setup.
            raise NotImplementedError(
                "Data parallel is not supported for multi-encoder setup."
            )

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    if args.train_dtype in ("float16", "float32", "float64"):
        dtype = getattr(torch, args.train_dtype)
    else:
        dtype = torch.float32
    model = model.to(device=device, dtype=dtype)

    if args.freeze_mods:
        model, model_params = freeze_modules(model, args.freeze_mods)
    else:
        model_params = model.parameters()

    # Setup an optimizer
    if args.opt == "adadelta":
        optimizer = torch.optim.Adadelta(
            model_params, rho=0.95, eps=args.eps, weight_decay=args.weight_decay
        )
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(model_params, weight_decay=args.weight_decay)
    elif args.opt == "noam":
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt

        # For transformer-transducer, adim declaration is within the block definition.
        # Thus, we need retrieve the most dominant value (d_hidden) for Noam scheduler.
        if hasattr(args, "enc_block_arch") or hasattr(args, "dec_block_arch"):
            adim = model.most_dom_dim
        else:
            adim = args.adim

        optimizer = get_std_opt(
            model_params, adim, args.transformer_warmup_steps, args.transformer_lr
        )
    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)

    # setup apex.amp
    if args.train_dtype in ("O0", "O1", "O2", "O3"):
        try:
            from apex import amp
        except ImportError as e:
            logging.error(
                f"You need to install apex for --train-dtype {args.train_dtype}. "
                "See https://github.com/NVIDIA/apex#linux"
            )
            raise e
        if args.opt == "noam":
            model, optimizer.optimizer = amp.initialize(
                model, optimizer.optimizer, opt_level=args.train_dtype
            )
        else:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=args.train_dtype
            )
        use_apex = True

        from espnet.nets.pytorch_backend.ctc import CTC

        amp.register_float_function(CTC, "loss_fn")
        amp.init()
        logging.warning("register ctc as float function")
    else:
        use_apex = False

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # Setup a converter
    if args.num_encs == 1:
        converter = CustomConverter(subsampling_factor=model.subsample[0], dtype=dtype)
    else:
        converter = CustomConverterMulEnc(
            [i[0] for i in model.subsample_list], dtype=dtype
        )

    # read json data
    with open(args.train_json, "rb") as f:
        train_json = json.load(f)["utts"]
    with open(args.valid_json, "rb") as f:
        valid_json = json.load(f)["utts"]

    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    # make minibatch list (variable length)
    train = make_batchset(
        train_json,
        args.batch_size,
        args.maxlen_in,
        args.maxlen_out,
        args.minibatches,
        min_batch_size=args.ngpu if args.ngpu > 1 else 1,
        shortest_first=use_sortagrad,
        count=args.batch_count,
        batch_bins=args.batch_bins,
        batch_frames_in=args.batch_frames_in,
        batch_frames_out=args.batch_frames_out,
        batch_frames_inout=args.batch_frames_inout,
        iaxis=0,
        oaxis=0,
    )
    valid = make_batchset(
        valid_json,
        args.batch_size,
        args.maxlen_in,
        args.maxlen_out,
        args.minibatches,
        min_batch_size=args.ngpu if args.ngpu > 1 else 1,
        count=args.batch_count,
        batch_bins=args.batch_bins,
        batch_frames_in=args.batch_frames_in,
        batch_frames_out=args.batch_frames_out,
        batch_frames_inout=args.batch_frames_inout,
        iaxis=0,
        oaxis=0,
    )

    load_tr = LoadInputsAndTargets(
        mode="asr",
        load_output=True,
        preprocess_conf=args.preprocess_conf,
        preprocess_args={"train": True},  # Switch the mode of preprocessing
    )
    load_cv = LoadInputsAndTargets(
        mode="asr",
        load_output=True,
        preprocess_conf=args.preprocess_conf,
        preprocess_args={"train": False},  # Switch the mode of preprocessing
    )
    # hack to make batchsize argument as 1
    # actual bathsize is included in a list
    # default collate function converts numpy array to pytorch tensor
    # we used an empty collate function instead which returns list
    train_iter = ChainerDataLoader(
        dataset=TransformDataset(train, lambda data: converter([load_tr(data)])),
        batch_size=1,
        num_workers=args.n_iter_processes,
        shuffle=not use_sortagrad,
        collate_fn=lambda x: x[0],
    )
    valid_iter = ChainerDataLoader(
        dataset=TransformDataset(valid, lambda data: converter([load_cv(data)])),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x[0],
        num_workers=args.n_iter_processes,
    )

    # Set up a trainer
    updater = CustomUpdater(
        model,
        args.grad_clip,
        {"main": train_iter},
        optimizer,
        device,
        args.ngpu,
        args.grad_noise,
        args.accum_grad,
        use_apex=use_apex,
    )
    trainer = training.Trainer(updater, (args.epochs, "epoch"), out=args.outdir)

    if use_sortagrad:
        trainer.extend(
            ShufflingEnabler([train_iter]),
            trigger=(args.sortagrad if args.sortagrad != -1 else args.epochs, "epoch"),
        )

    # Resume from a snapshot
    if args.resume:
        logging.info("resumed from %s" % args.resume)
        torch_resume(args.resume, trainer)

    # Evaluate the model with the test dataset for each epoch
    if args.save_interval_iters > 0:
        trainer.extend(
            CustomEvaluator(model, {"main": valid_iter}, reporter, device, args.ngpu),
            trigger=(args.save_interval_iters, "iteration"),
        )
    else:
        trainer.extend(
            CustomEvaluator(model, {"main": valid_iter}, reporter, device, args.ngpu)
        )

    # Save attention weight each epoch
    is_attn_plot = (
        (
            "transformer" in args.model_module
            or "conformer" in args.model_module
            or mtl_mode in ["att", "mtl"]
        )
        or (
            mtl_mode == "transducer" and getattr(args, "rnnt_mode", False) == "rnnt-att"
        )
        or mtl_mode == "transformer_transducer"
    )

    if args.num_save_attention > 0 and is_attn_plot:
        data = sorted(
            list(valid_json.items())[: args.num_save_attention],
            key=lambda x: int(x[1]["input"][0]["shape"][1]),
            reverse=True,
        )
        if hasattr(model, "module"):
            att_vis_fn = model.module.calculate_all_attentions
            plot_class = model.module.attention_plot_class
        else:
            att_vis_fn = model.calculate_all_attentions
            plot_class = model.attention_plot_class
        att_reporter = plot_class(
            att_vis_fn,
            data,
            args.outdir + "/att_ws",
            converter=converter,
            transform=load_cv,
            device=device,
        )
        trainer.extend(att_reporter, trigger=(1, "epoch"))
    else:
        att_reporter = None

    # Save CTC prob at each epoch
    if mtl_mode in ["ctc", "mtl"] and args.num_save_ctc > 0:
        # NOTE: sort it by output lengths
        data = sorted(
            list(valid_json.items())[: args.num_save_ctc],
            key=lambda x: int(x[1]["output"][0]["shape"][0]),
            reverse=True,
        )
        if hasattr(model, "module"):
            ctc_vis_fn = model.module.calculate_all_ctc_probs
            plot_class = model.module.ctc_plot_class
        else:
            ctc_vis_fn = model.calculate_all_ctc_probs
            plot_class = model.ctc_plot_class
        ctc_reporter = plot_class(
            ctc_vis_fn,
            data,
            args.outdir + "/ctc_prob",
            converter=converter,
            transform=load_cv,
            device=device,
            ikey="output",
            iaxis=1,
        )
        trainer.extend(ctc_reporter, trigger=(1, "epoch"))
    else:
        ctc_reporter = None

    # Make a plot for training and validation values
    if args.num_encs > 1:
        report_keys_loss_ctc = [
            "main/loss_ctc{}".format(i + 1) for i in range(model.num_encs)
        ] + ["validation/main/loss_ctc{}".format(i + 1) for i in range(model.num_encs)]
        report_keys_cer_ctc = [
            "main/cer_ctc{}".format(i + 1) for i in range(model.num_encs)
        ] + ["validation/main/cer_ctc{}".format(i + 1) for i in range(model.num_encs)]
    trainer.extend(
        extensions.PlotReport(
            [
                "main/loss",
                "validation/main/loss",
                "main/loss_ctc",
                "validation/main/loss_ctc",
                "main/loss_att",
                "validation/main/loss_att",
            ]
            + ([] if args.num_encs == 1 else report_keys_loss_ctc),
            "epoch",
            file_name="loss.png",
        )
    )
    trainer.extend(
        extensions.PlotReport(
            ["main/acc", "validation/main/acc"], "epoch", file_name="acc.png"
        )
    )
    trainer.extend(
        extensions.PlotReport(
            ["main/cer_ctc", "validation/main/cer_ctc"]
            + ([] if args.num_encs == 1 else report_keys_loss_ctc),
            "epoch",
            file_name="cer.png",
        )
    )

    # Save best models
    trainer.extend(
        snapshot_object(model, "model.loss.best"),
        trigger=training.triggers.MinValueTrigger("validation/main/loss"),
    )
    if mtl_mode not in ["ctc", "transducer", "transformer_transducer"]:
        trainer.extend(
            snapshot_object(model, "model.acc.best"),
            trigger=training.triggers.MaxValueTrigger("validation/main/acc"),
        )

    # save snapshot which contains model and optimizer states
    if args.save_interval_iters > 0:
        trainer.extend(
            torch_snapshot(filename="snapshot.iter.{.updater.iteration}"),
            trigger=(args.save_interval_iters, "iteration"),
        )
    else:
        trainer.extend(torch_snapshot(), trigger=(1, "epoch"))

    # epsilon decay in the optimizer
    if args.opt == "adadelta":
        if args.criterion == "acc" and mtl_mode != "ctc":
            trainer.extend(
                restore_snapshot(
                    model, args.outdir + "/model.acc.best", load_fn=torch_load
                ),
                trigger=CompareValueTrigger(
                    "validation/main/acc",
                    lambda best_value, current_value: best_value > current_value,
                ),
            )
            trainer.extend(
                adadelta_eps_decay(args.eps_decay),
                trigger=CompareValueTrigger(
                    "validation/main/acc",
                    lambda best_value, current_value: best_value > current_value,
                ),
            )
        elif args.criterion == "loss":
            trainer.extend(
                restore_snapshot(
                    model, args.outdir + "/model.loss.best", load_fn=torch_load
                ),
                trigger=CompareValueTrigger(
                    "validation/main/loss",
                    lambda best_value, current_value: best_value < current_value,
                ),
            )
            trainer.extend(
                adadelta_eps_decay(args.eps_decay),
                trigger=CompareValueTrigger(
                    "validation/main/loss",
                    lambda best_value, current_value: best_value < current_value,
                ),
            )
        # NOTE: In some cases, it may take more than one epoch for the model's loss
        # to escape from a local minimum.
        # Thus, restore_snapshot extension is not used here.
        # see details in https://github.com/espnet/espnet/pull/2171
        elif args.criterion == "loss_eps_decay_only":
            trainer.extend(
                adadelta_eps_decay(args.eps_decay),
                trigger=CompareValueTrigger(
                    "validation/main/loss",
                    lambda best_value, current_value: best_value < current_value,
                ),
            )

    # Write a log of evaluation statistics for each epoch
    trainer.extend(
        extensions.LogReport(trigger=(args.report_interval_iters, "iteration"))
    )
    report_keys = [
        "epoch",
        "iteration",
        "main/loss",
        "main/loss_ctc",
        "main/loss_att",
        "validation/main/loss",
        "validation/main/loss_ctc",
        "validation/main/loss_att",
        "main/acc",
        "validation/main/acc",
        "main/cer_ctc",
        "validation/main/cer_ctc",
        "elapsed_time",
    ] + ([] if args.num_encs == 1 else report_keys_cer_ctc + report_keys_loss_ctc)
    if args.opt == "adadelta":
        trainer.extend(
            extensions.observe_value(
                "eps",
                lambda trainer: trainer.updater.get_optimizer("main").param_groups[0][
                    "eps"
                ],
            ),
            trigger=(args.report_interval_iters, "iteration"),
        )
        report_keys.append("eps")
    if args.report_cer:
        report_keys.append("validation/main/cer")
    if args.report_wer:
        report_keys.append("validation/main/wer")
    trainer.extend(
        extensions.PrintReport(report_keys),
        trigger=(args.report_interval_iters, "iteration"),
    )

    trainer.extend(extensions.ProgressBar(update_interval=args.report_interval_iters))
    set_early_stop(trainer, args)

    if args.tensorboard_dir is not None and args.tensorboard_dir != "":
        trainer.extend(
            TensorboardLogger(
                SummaryWriter(args.tensorboard_dir),
                att_reporter=att_reporter,
                ctc_reporter=ctc_reporter,
            ),
            trigger=(args.report_interval_iters, "iteration"),
        )
    # Run the training
    trainer.run()
    check_early_stop(trainer, args.epochs)
