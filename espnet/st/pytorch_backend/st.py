# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Training/decoding definition for the speech translation task."""

import itertools
import json
import logging
import os

import numpy as np
import torch
from chainer import training
from chainer.training import extensions

from espnet.asr.asr_utils import (
    CompareValueTrigger,
    adadelta_eps_decay,
    adam_lr_decay,
    add_results_to_json,
    restore_snapshot,
    snapshot_object,
    torch_load,
    torch_resume,
    torch_snapshot,
)
from espnet.asr.pytorch_backend.asr import CustomConverter as ASRCustomConverter
from espnet.asr.pytorch_backend.asr import CustomEvaluator, CustomUpdater
from espnet.asr.pytorch_backend.asr_init import load_trained_model, load_trained_modules
from espnet.nets.pytorch_backend.e2e_asr import pad_list
from espnet.nets.st_interface import STInterface
from espnet.utils.dataset import ChainerDataLoader, TransformDataset
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.tensorboard_logger import TensorboardLogger
from espnet.utils.training.train_utils import check_early_stop, set_early_stop


class CustomConverter(ASRCustomConverter):
    """Custom batch converter for Pytorch.

    Args:
        subsampling_factor (int): The subsampling factor.
        dtype (torch.dtype): Data type to convert.
        use_source_text (bool): use source transcription.

    """

    def __init__(
        self, subsampling_factor=1, dtype=torch.float32, use_source_text=False
    ):
        """Construct a CustomConverter object."""
        super().__init__(subsampling_factor=subsampling_factor, dtype=dtype)
        self.use_source_text = use_source_text

    def __call__(self, batch, device=torch.device("cpu")):
        """Transform a batch and send it to a device.

        Args:
            batch (list): The batch to transform.
            device (torch.device): The device to send to.

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor)

        """
        # batch should be located in list
        assert len(batch) == 1
        xs, ys, ys_src = batch[0]

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs])
        ilens = torch.from_numpy(ilens).to(device)

        xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0).to(
            device, dtype=self.dtype
        )

        ys_pad = pad_list(
            [torch.from_numpy(np.array(y, dtype=np.int64)) for y in ys],
            self.ignore_id,
        ).to(device)

        if self.use_source_text:
            ys_pad_src = pad_list(
                [torch.from_numpy(np.array(y, dtype=np.int64)) for y in ys_src],
                self.ignore_id,
            ).to(device)
        else:
            ys_pad_src = None

        return xs_pad, ilens, ys_pad, ys_pad_src


def train(args):
    """Train with the given args.

    Args:
        args (namespace): The program arguments.

    """
    set_deterministic_pytorch(args)

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning("cuda is not available")

    # get input and output dimension info
    with open(args.valid_json, "rb") as f:
        valid_json = json.load(f)["utts"]
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]["input"][0]["shape"][-1])
    odim = int(valid_json[utts[0]]["output"][0]["shape"][-1])
    logging.info("#input dims : " + str(idim))
    logging.info("#output dims: " + str(odim))

    # Initialize with pre-trained ASR encoder and MT decoder
    if args.enc_init is not None or args.dec_init is not None:
        model = load_trained_modules(idim, odim, args, interface=STInterface)
    else:
        model_class = dynamic_import(args.model_module)
        model = model_class(idim, odim, args)
    assert isinstance(model, STInterface)
    total_subsampling_factor = model.get_total_subsampling_factor()

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + "/model.json"
    with open(model_conf, "wb") as f:
        logging.info("writing a model config file to " + model_conf)
        f.write(
            json.dumps(
                (idim, odim, vars(args)), indent=4, ensure_ascii=False, sort_keys=True
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

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    if args.train_dtype in ("float16", "float32", "float64"):
        dtype = getattr(torch, args.train_dtype)
    else:
        dtype = torch.float32
    model = model.to(device=device, dtype=dtype)

    logging.warning(
        "num. model params: {:,} (num. trained: {:,} ({:.1f}%))".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.parameters() if p.requires_grad)
            * 100.0
            / sum(p.numel() for p in model.parameters()),
        )
    )

    # Setup an optimizer
    if args.opt == "adadelta":
        optimizer = torch.optim.Adadelta(
            model.parameters(), rho=0.95, eps=args.eps, weight_decay=args.weight_decay
        )
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.opt == "noam":
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt

        optimizer = get_std_opt(
            model.parameters(),
            args.adim,
            args.transformer_warmup_steps,
            args.transformer_lr,
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
    else:
        use_apex = False

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # Setup a converter
    converter = CustomConverter(
        subsampling_factor=model.subsample[0],
        dtype=dtype,
        use_source_text=args.asr_weight > 0 or args.mt_weight > 0,
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

    # Save attention weight at each epoch
    if args.num_save_attention > 0:
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
            subsampling_factor=total_subsampling_factor,
        )
        trainer.extend(att_reporter, trigger=(1, "epoch"))
    else:
        att_reporter = None

    # Save CTC prob at each epoch
    if (args.asr_weight > 0 and args.mtlalpha > 0) and args.num_save_ctc > 0:
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
            subsampling_factor=total_subsampling_factor,
        )
        trainer.extend(ctc_reporter, trigger=(1, "epoch"))
    else:
        ctc_reporter = None

    # Make a plot for training and validation values
    trainer.extend(
        extensions.PlotReport(
            [
                "main/loss",
                "validation/main/loss",
                "main/loss_asr",
                "validation/main/loss_asr",
                "main/loss_mt",
                "validation/main/loss_mt",
                "main/loss_st",
                "validation/main/loss_st",
            ],
            "epoch",
            file_name="loss.png",
        )
    )
    trainer.extend(
        extensions.PlotReport(
            [
                "main/acc",
                "validation/main/acc",
                "main/acc_asr",
                "validation/main/acc_asr",
                "main/acc_mt",
                "validation/main/acc_mt",
            ],
            "epoch",
            file_name="acc.png",
        )
    )
    trainer.extend(
        extensions.PlotReport(
            ["main/bleu", "validation/main/bleu"], "epoch", file_name="bleu.png"
        )
    )

    # Save best models
    trainer.extend(
        snapshot_object(model, "model.loss.best"),
        trigger=training.triggers.MinValueTrigger("validation/main/loss"),
    )
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
        if args.criterion == "acc":
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
    elif args.opt == "adam":
        if args.criterion == "acc":
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
                adam_lr_decay(args.lr_decay),
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
                adam_lr_decay(args.lr_decay),
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
        "main/loss_st",
        "main/loss_asr",
        "validation/main/loss",
        "validation/main/loss_st",
        "validation/main/loss_asr",
        "main/acc",
        "validation/main/acc",
    ]
    if args.asr_weight > 0:
        report_keys.append("main/acc_asr")
        report_keys.append("validation/main/acc_asr")
    report_keys += ["elapsed_time"]
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
    elif args.opt in ["adam", "noam"]:
        trainer.extend(
            extensions.observe_value(
                "lr",
                lambda trainer: trainer.updater.get_optimizer("main").param_groups[0][
                    "lr"
                ],
            ),
            trigger=(args.report_interval_iters, "iteration"),
        )
        report_keys.append("lr")
    if args.asr_weight > 0:
        if args.mtlalpha > 0:
            report_keys.append("main/cer_ctc")
            report_keys.append("validation/main/cer_ctc")
        if args.mtlalpha < 1:
            if args.report_cer:
                report_keys.append("validation/main/cer")
            if args.report_wer:
                report_keys.append("validation/main/wer")
    if args.report_bleu:
        report_keys.append("main/bleu")
        report_keys.append("validation/main/bleu")
    trainer.extend(
        extensions.PrintReport(report_keys),
        trigger=(args.report_interval_iters, "iteration"),
    )

    trainer.extend(extensions.ProgressBar(update_interval=args.report_interval_iters))
    set_early_stop(trainer, args)

    if args.tensorboard_dir is not None and args.tensorboard_dir != "":
        from torch.utils.tensorboard import SummaryWriter

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


def trans(args):
    """Decode with the given args.

    Args:
        args (namespace): The program arguments.

    """
    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model)
    assert isinstance(model, STInterface)
    model.trans_args = args

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info("gpu id: " + str(gpu_id))
        model.cuda()

    # read json data
    with open(args.trans_json, "rb") as f:
        js = json.load(f)["utts"]
    new_js = {}

    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=False,
        sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None
        else args.preprocess_conf,
        preprocess_args={"train": False},
    )

    if args.batchsize == 0:
        with torch.no_grad():
            for idx, name in enumerate(js.keys(), 1):
                logging.info("(%d/%d) decoding " + name, idx, len(js.keys()))
                batch = [(name, js[name])]
                feat = load_inputs_and_targets(batch)[0][0]
                nbest_hyps = model.translate(
                    feat,
                    args,
                    train_args.char_list,
                )
                new_js[name] = add_results_to_json(
                    js[name], nbest_hyps, train_args.char_list
                )

    else:

        def grouper(n, iterable, fillvalue=None):
            kargs = [iter(iterable)] * n
            return itertools.zip_longest(*kargs, fillvalue=fillvalue)

        # sort data if batchsize > 1
        keys = list(js.keys())
        if args.batchsize > 1:
            feat_lens = [js[key]["input"][0]["shape"][0] for key in keys]
            sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
            keys = [keys[i] for i in sorted_index]

        with torch.no_grad():
            for names in grouper(args.batchsize, keys, None):
                names = [name for name in names if name]
                batch = [(name, js[name]) for name in names]
                feats = load_inputs_and_targets(batch)[0]
                nbest_hyps = model.translate_batch(
                    feats,
                    args,
                    train_args.char_list,
                )

                for i, nbest_hyp in enumerate(nbest_hyps):
                    name = names[i]
                    new_js[name] = add_results_to_json(
                        js[name], nbest_hyp, train_args.char_list
                    )

    with open(args.result_label, "wb") as f:
        f.write(
            json.dumps(
                {"utts": new_js}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )
