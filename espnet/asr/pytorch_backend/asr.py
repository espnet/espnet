# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Training/decoding definition for the speech recognition task."""

import copy
import itertools
import json
import logging
import math
import os

import numpy as np
import torch
from chainer import reporter as reporter_module
from chainer import training
from chainer.training import extensions
from chainer.training.updater import StandardUpdater
from packaging.version import parse as V
from torch.nn.parallel import data_parallel

import espnet.lm.pytorch_backend.extlm as extlm_pytorch
import espnet.nets.pytorch_backend.lm.default as lm_pytorch
from espnet.asr.asr_utils import (
    CompareValueTrigger,
    adadelta_eps_decay,
    add_results_to_json,
    format_mulenc_args,
    get_model_conf,
    plot_spectrogram,
    restore_snapshot,
    snapshot_object,
    torch_load,
    torch_resume,
    torch_snapshot,
)
from espnet.asr.pytorch_backend.asr_init import (
    freeze_modules,
    load_trained_model,
    load_trained_modules,
)
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.beam_search_transducer import BeamSearchTransducer
from espnet.nets.pytorch_backend.e2e_asr import pad_list
from espnet.nets.pytorch_backend.streaming.segment import SegmentStreamingE2E
from espnet.nets.pytorch_backend.streaming.window import WindowStreamingE2E
from espnet.transform.spectrogram import IStft
from espnet.transform.transformation import Transformation
from espnet.utils.cli_writers import file_writer_helper
from espnet.utils.dataset import ChainerDataLoader, TransformDataset
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.training.evaluator import BaseEvaluator
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.tensorboard_logger import TensorboardLogger
from espnet.utils.training.train_utils import check_early_stop, set_early_stop


def _recursive_to(xs, device):
    if torch.is_tensor(xs):
        return xs.to(device)
    if isinstance(xs, tuple):
        return tuple(_recursive_to(x, device) for x in xs)
    return xs


class CustomEvaluator(BaseEvaluator):
    """Custom Evaluator for Pytorch.

    Args:
        model (torch.nn.Module): The model to evaluate.
        iterator (chainer.dataset.Iterator) : The train iterator.

        target (link | dict[str, link]) :Link object or a dictionary of
            links to evaluate. If this is just a link object, the link is
            registered by the name ``'main'``.

        device (torch.device): The device used.
        ngpu (int): The number of GPUs.

    """

    def __init__(self, model, iterator, target, device, ngpu=None):
        super(CustomEvaluator, self).__init__(iterator, target)
        self.model = model
        self.device = device
        if ngpu is not None:
            self.ngpu = ngpu
        elif device.type == "cpu":
            self.ngpu = 0
        else:
            self.ngpu = 1

    # The core part of the update routine can be customized by overriding
    def evaluate(self):
        """Main evaluate routine for CustomEvaluator."""
        iterator = self._iterators["main"]

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, "reset"):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        self.model.eval()
        with torch.no_grad():
            for batch in it:
                x = _recursive_to(batch, self.device)
                observation = {}
                with reporter_module.report_scope(observation):
                    # read scp files
                    # x: original json with loaded features
                    #    will be converted to chainer variable later
                    if self.ngpu == 0:
                        self.model(*x)
                    else:
                        # apex does not support torch.nn.DataParallel
                        data_parallel(self.model, x, range(self.ngpu))

                summary.add(observation)
        self.model.train()

        return summary.compute_mean()


class CustomUpdater(StandardUpdater):
    """Custom Updater for Pytorch.

    Args:
        model (torch.nn.Module): The model to update.
        grad_clip_threshold (float): The gradient clipping value to use.
        train_iter (chainer.dataset.Iterator): The training iterator.
        optimizer (torch.optim.optimizer): The training optimizer.

        device (torch.device): The device to use.
        ngpu (int): The number of gpus to use.
        use_apex (bool): The flag to use Apex in backprop.

    """

    def __init__(
        self,
        model,
        grad_clip_threshold,
        train_iter,
        optimizer,
        device,
        ngpu,
        grad_noise=False,
        accum_grad=1,
        use_apex=False,
    ):
        super(CustomUpdater, self).__init__(train_iter, optimizer)
        self.model = model
        self.grad_clip_threshold = grad_clip_threshold
        self.device = device
        self.ngpu = ngpu
        self.accum_grad = accum_grad
        self.forward_count = 0
        self.grad_noise = grad_noise
        self.iteration = 0
        self.use_apex = use_apex

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        """Main update routine of the CustomUpdater."""
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator("main")
        optimizer = self.get_optimizer("main")
        epoch = train_iter.epoch

        # Get the next batch (a list of json files)
        batch = train_iter.next()
        # self.iteration += 1 # Increase may result in early report,
        # which is done in other place automatically.
        x = _recursive_to(batch, self.device)
        is_new_epoch = train_iter.epoch != epoch
        # When the last minibatch in the current epoch is given,
        # gradient accumulation is turned off in order to evaluate the model
        # on the validation set in every epoch.
        # see details in https://github.com/espnet/espnet/pull/1388

        # Compute the loss at this time step and accumulate it
        if self.ngpu == 0:
            loss = self.model(*x).mean() / self.accum_grad
        else:
            # apex does not support torch.nn.DataParallel
            loss = (
                data_parallel(self.model, x, range(self.ngpu)).mean() / self.accum_grad
            )
        if self.use_apex:
            from apex import amp

            # NOTE: for a compatibility with noam optimizer
            opt = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # gradient noise injection
        if self.grad_noise:
            from espnet.asr.asr_utils import add_gradient_noise

            add_gradient_noise(
                self.model, self.iteration, duration=100, eta=1.0, scale_factor=0.55
            )

        # update parameters
        self.forward_count += 1
        if not is_new_epoch and self.forward_count != self.accum_grad:
            return
        self.forward_count = 0
        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_threshold
        )
        logging.info("grad norm={}".format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning("grad norm is nan. Do not update model.")
        else:
            optimizer.step()
        optimizer.zero_grad()

    def update(self):
        self.update_core()
        # #iterations with accum_grad > 1
        # Ref.: https://github.com/espnet/espnet/issues/777
        if self.forward_count == 0:
            self.iteration += 1


class CustomConverter(object):
    """Custom batch converter for Pytorch.

    Args:
        subsampling_factor (int): The subsampling factor.
        dtype (torch.dtype): Data type to convert.

    """

    def __init__(self, subsampling_factor=1, dtype=torch.float32):
        """Construct a CustomConverter object."""
        self.subsampling_factor = subsampling_factor
        self.ignore_id = -1
        self.dtype = dtype

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
        xs, ys = batch[0]

        # perform subsampling
        if self.subsampling_factor > 1:
            xs = [x[:: self.subsampling_factor, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs])

        # perform padding and convert to tensor
        # currently only support real number
        if xs[0].dtype.kind == "c":
            xs_pad_real = pad_list(
                [torch.from_numpy(x.real).float() for x in xs], 0
            ).to(device, dtype=self.dtype)
            xs_pad_imag = pad_list(
                [torch.from_numpy(x.imag).float() for x in xs], 0
            ).to(device, dtype=self.dtype)
            # Note(kamo):
            # {'real': ..., 'imag': ...} will be changed to ComplexTensor in E2E.
            # Don't create ComplexTensor and give it E2E here
            # because torch.nn.DataParellel can't handle it.
            xs_pad = {"real": xs_pad_real, "imag": xs_pad_imag}
        else:
            xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0).to(
                device, dtype=self.dtype
            )

        ilens = torch.from_numpy(ilens).to(device)
        # NOTE: this is for multi-output (e.g., speech translation)
        ys_pad = pad_list(
            [
                torch.from_numpy(
                    np.array(y[0][:]) if isinstance(y, tuple) else y
                ).long()
                for y in ys
            ],
            self.ignore_id,
        ).to(device)

        return xs_pad, ilens, ys_pad


class CustomConverterMulEnc(object):
    """Custom batch converter for Pytorch in multi-encoder case.

    Args:
        subsampling_factors (list): List of subsampling factors for each encoder.
        dtype (torch.dtype): Data type to convert.

    """

    def __init__(self, subsampling_factors=[1, 1], dtype=torch.float32):
        """Initialize the converter."""
        self.subsampling_factors = subsampling_factors
        self.ignore_id = -1
        self.dtype = dtype
        self.num_encs = len(subsampling_factors)

    def __call__(self, batch, device=torch.device("cpu")):
        """Transform a batch and send it to a device.

        Args:
            batch (list): The batch to transform.
            device (torch.device): The device to send to.

        Returns:
            tuple( list(torch.Tensor), list(torch.Tensor), torch.Tensor)

        """
        # batch should be located in list
        assert len(batch) == 1
        xs_list = batch[0][: self.num_encs]
        ys = batch[0][-1]

        # perform subsampling
        if np.sum(self.subsampling_factors) > self.num_encs:
            xs_list = [
                [x[:: self.subsampling_factors[i], :] for x in xs_list[i]]
                for i in range(self.num_encs)
            ]

        # get batch of lengths of input sequences
        ilens_list = [
            np.array([x.shape[0] for x in xs_list[i]]) for i in range(self.num_encs)
        ]

        # perform padding and convert to tensor
        # currently only support real number
        xs_list_pad = [
            pad_list([torch.from_numpy(x).float() for x in xs_list[i]], 0).to(
                device, dtype=self.dtype
            )
            for i in range(self.num_encs)
        ]

        ilens_list = [
            torch.from_numpy(ilens_list[i]).to(device) for i in range(self.num_encs)
        ]
        # NOTE: this is for multi-task learning (e.g., speech translation)
        ys_pad = pad_list(
            [
                torch.from_numpy(np.array(y[0]) if isinstance(y, tuple) else y).long()
                for y in ys
            ],
            self.ignore_id,
        ).to(device)

        return xs_list_pad, ilens_list, ys_pad


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
    for i in range(args.num_encs):
        logging.info("stream{}: input dims : {}".format(i + 1, idim_list[i]))
    logging.info("#output dims: " + str(odim))

    # specify attention, CTC, hybrid mode
    if "transducer" in args.model_module:
        if (
            getattr(args, "etype", False) == "custom"
            or getattr(args, "dtype", False) == "custom"
        ):
            mtl_mode = "custom_transducer"
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
    total_subsampling_factor = model.get_total_subsampling_factor()

    logging.info(
        " Total parameter of the model = "
        + str(sum(p.numel() for p in model.parameters()))
    )

    if args.rnnlm is not None:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(len(args.char_list), rnnlm_args.layer, rnnlm_args.unit)
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
            model_params, rho=0.95, eps=args.eps, weight_decay=args.weight_decay
        )
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(model_params, weight_decay=args.weight_decay)
    elif args.opt == "noam":
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt

        if "transducer" in mtl_mode:
            if args.noam_adim > 0:
                optimizer = get_std_opt(
                    model_params,
                    args.noam_adim,
                    args.optimizer_warmup_steps,
                    args.noam_lr,
                )
            else:
                raise ValueError("noam-adim option should be set to use Noam scheduler")
        else:
            optimizer = get_std_opt(
                model_params,
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
        "transformer" in args.model_module
        or "conformer" in args.model_module
        or mtl_mode in ["att", "mtl", "custom_transducer"]
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
            subsampling_factor=total_subsampling_factor,
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
            subsampling_factor=total_subsampling_factor,
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

    if hasattr(model, "is_transducer"):
        trans_keys = [
            "main/loss",
            "validation/main/loss",
            "main/loss_trans",
            "validation/main/loss_trans",
        ]

        ctc_keys = (
            ["main/loss_ctc", "validation/main/loss_ctc"] if args.use_ctc_loss else []
        )

        aux_trans_keys = (
            ["main/loss_aux_trans", "validation/main/loss_aux_trans",]
            if args.use_aux_transducer_loss
            else []
        )

        symm_kl_div_keys = (
            ["main/loss_symm_kl_div", "validation/main/loss_symm_kl_div",]
            if args.use_symm_kl_div_loss
            else []
        )

        lm_keys = (
            ["main/loss_lm", "validation/main/loss_lm",] if args.use_lm_loss else []
        )

        transducer_keys = (
            trans_keys + ctc_keys + aux_trans_keys + symm_kl_div_keys + lm_keys
        )

        trainer.extend(
            extensions.PlotReport(transducer_keys, "epoch", file_name="loss.png",)
        )
    else:
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
    if mtl_mode not in ["ctc", "transducer", "custom_transducer"]:
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

    # save snapshot at every epoch - for model averaging
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

    if hasattr(model, "is_transducer"):
        report_keys = ["epoch", "iteration",] + transducer_keys + ["elapsed_time"]
    else:
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


def recog(args):
    """Decode with the given args.

    Args:
        args (namespace): The program arguments.

    """
    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model, training=False)
    assert isinstance(model, ASRInterface)
    model.recog_args = args

    if args.quantize_config is not None:
        q_config = set([getattr(torch.nn, q) for q in args.quantize_config])
    else:
        q_config = {torch.nn.Linear}

    if args.quantize_asr_model:
        logging.info("Use a quantized ASR model for decoding.")

        # It seems quantized LSTM only supports non-packed sequence before torch 1.4.0.
        # Reference issue: https://github.com/pytorch/pytorch/issues/27963
        if (
            V(torch.__version__) < V("1.4.0")
            and "lstm" in train_args.etype
            and torch.nn.LSTM in q_config
        ):
            raise ValueError(
                "Quantized LSTM in ESPnet is only supported with torch 1.4+."
            )

        # Dunno why but weight_observer from dynamic quantized module must have
        # dtype=torch.qint8 with torch < 1.5 although dtype=torch.float16 is supported.
        if args.quantize_dtype == "float16" and V(torch.__version__) < V("1.5.0"):
            raise ValueError(
                "float16 dtype for dynamic quantization is not supported with torch "
                "version < 1.5.0. Switching to qint8 dtype instead."
            )

        dtype = getattr(torch, args.quantize_dtype)

        model = torch.quantization.quantize_dynamic(model, q_config, dtype=dtype)

    if args.streaming_mode and "transformer" in train_args.model_module:
        raise NotImplementedError("streaming mode for transformer is not implemented")
    logging.info(
        " Total parameter of the model = "
        + str(sum(p.numel() for p in model.parameters()))
    )

    # read rnnlm
    if args.rnnlm:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        if getattr(rnnlm_args, "model_module", "default") != "default":
            raise ValueError(
                "use '--api v2' option to decode with non-default language model"
            )
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(train_args.char_list),
                rnnlm_args.layer,
                rnnlm_args.unit,
                getattr(rnnlm_args, "embed_unit", None),  # for backward compatibility
            )
        )
        torch_load(args.rnnlm, rnnlm)
        if args.quantize_lm_model:
            dtype = getattr(torch, args.quantize_dtype)
            rnnlm = torch.quantization.quantize_dynamic(rnnlm, q_config, dtype=dtype)
        rnnlm.eval()
    else:
        rnnlm = None

    if args.word_rnnlm:
        rnnlm_args = get_model_conf(args.word_rnnlm, args.word_rnnlm_conf)
        word_dict = rnnlm_args.char_list_dict
        char_dict = {x: i for i, x in enumerate(train_args.char_list)}
        word_rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(word_dict),
                rnnlm_args.layer,
                rnnlm_args.unit,
                getattr(rnnlm_args, "embed_unit", None),  # for backward compatibility
            )
        )
        torch_load(args.word_rnnlm, word_rnnlm)
        word_rnnlm.eval()

        if rnnlm is not None:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.MultiLevelLM(
                    word_rnnlm.predictor, rnnlm.predictor, word_dict, char_dict
                )
            )
        else:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.LookAheadWordLM(
                    word_rnnlm.predictor, word_dict, char_dict
                )
            )

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info("gpu id: " + str(gpu_id))
        model.cuda()
        if rnnlm:
            rnnlm.cuda()

    # read json data
    with open(args.recog_json, "rb") as f:
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

    # load transducer beam search
    if hasattr(model, "is_transducer"):
        if hasattr(model, "dec"):
            trans_decoder = model.dec
        else:
            trans_decoder = model.decoder
        joint_network = model.transducer_tasks.joint_network

        beam_search_transducer = BeamSearchTransducer(
            decoder=trans_decoder,
            joint_network=joint_network,
            beam_size=args.beam_size,
            lm=rnnlm,
            lm_weight=args.lm_weight,
            search_type=args.search_type,
            max_sym_exp=args.max_sym_exp,
            u_max=args.u_max,
            nstep=args.nstep,
            prefix_alpha=args.prefix_alpha,
            expansion_gamma=args.expansion_gamma,
            expansion_beta=args.expansion_beta,
            score_norm=args.score_norm,
            softmax_temperature=args.softmax_temperature,
            nbest=args.nbest,
            quantization=args.quantize_asr_model,
        )

    if args.batchsize == 0:
        with torch.no_grad():
            for idx, name in enumerate(js.keys(), 1):
                logging.info("(%d/%d) decoding " + name, idx, len(js.keys()))
                batch = [(name, js[name])]
                feat = load_inputs_and_targets(batch)
                feat = (
                    feat[0][0]
                    if args.num_encs == 1
                    else [feat[idx][0] for idx in range(model.num_encs)]
                )
                if args.streaming_mode == "window" and args.num_encs == 1:
                    logging.info(
                        "Using streaming recognizer with window size %d frames",
                        args.streaming_window,
                    )
                    se2e = WindowStreamingE2E(e2e=model, recog_args=args, rnnlm=rnnlm)
                    for i in range(0, feat.shape[0], args.streaming_window):
                        logging.info(
                            "Feeding frames %d - %d", i, i + args.streaming_window
                        )
                        se2e.accept_input(feat[i : i + args.streaming_window])
                    logging.info("Running offline attention decoder")
                    se2e.decode_with_attention_offline()
                    logging.info("Offline attention decoder finished")
                    nbest_hyps = se2e.retrieve_recognition()
                elif args.streaming_mode == "segment" and args.num_encs == 1:
                    logging.info(
                        "Using streaming recognizer with threshold value %d",
                        args.streaming_min_blank_dur,
                    )
                    nbest_hyps = []
                    for n in range(args.nbest):
                        nbest_hyps.append({"yseq": [], "score": 0.0})
                    se2e = SegmentStreamingE2E(e2e=model, recog_args=args, rnnlm=rnnlm)
                    r = np.prod(model.subsample)
                    for i in range(0, feat.shape[0], r):
                        hyps = se2e.accept_input(feat[i : i + r])
                        if hyps is not None:
                            text = "".join(
                                [
                                    train_args.char_list[int(x)]
                                    for x in hyps[0]["yseq"][1:-1]
                                    if int(x) != -1
                                ]
                            )
                            text = text.replace(
                                "\u2581", " "
                            ).strip()  # for SentencePiece
                            text = text.replace(model.space, " ")
                            text = text.replace(model.blank, "")
                            logging.info(text)
                            for n in range(args.nbest):
                                nbest_hyps[n]["yseq"].extend(hyps[n]["yseq"])
                                nbest_hyps[n]["score"] += hyps[n]["score"]
                elif hasattr(model, "is_transducer"):
                    nbest_hyps = model.recognize(feat, beam_search_transducer)
                else:
                    nbest_hyps = model.recognize(
                        feat, args, train_args.char_list, rnnlm
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
                feats = (
                    load_inputs_and_targets(batch)[0]
                    if args.num_encs == 1
                    else load_inputs_and_targets(batch)
                )
                if args.streaming_mode == "window" and args.num_encs == 1:
                    raise NotImplementedError
                elif args.streaming_mode == "segment" and args.num_encs == 1:
                    if args.batchsize > 1:
                        raise NotImplementedError
                    feat = feats[0]
                    nbest_hyps = []
                    for n in range(args.nbest):
                        nbest_hyps.append({"yseq": [], "score": 0.0})
                    se2e = SegmentStreamingE2E(e2e=model, recog_args=args, rnnlm=rnnlm)
                    r = np.prod(model.subsample)
                    for i in range(0, feat.shape[0], r):
                        hyps = se2e.accept_input(feat[i : i + r])
                        if hyps is not None:
                            text = "".join(
                                [
                                    train_args.char_list[int(x)]
                                    for x in hyps[0]["yseq"][1:-1]
                                    if int(x) != -1
                                ]
                            )
                            text = text.replace(
                                "\u2581", " "
                            ).strip()  # for SentencePiece
                            text = text.replace(model.space, " ")
                            text = text.replace(model.blank, "")
                            logging.info(text)
                            for n in range(args.nbest):
                                nbest_hyps[n]["yseq"].extend(hyps[n]["yseq"])
                                nbest_hyps[n]["score"] += hyps[n]["score"]
                    nbest_hyps = [nbest_hyps]
                else:
                    nbest_hyps = model.recognize_batch(
                        feats, args, train_args.char_list, rnnlm=rnnlm
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


def enhance(args):
    """Dumping enhanced speech and mask.

    Args:
        args (namespace): The program arguments.
    """
    set_deterministic_pytorch(args)
    # read training config
    idim, odim, train_args = get_model_conf(args.model, args.model_conf)

    # TODO(ruizhili): implement enhance for multi-encoder model
    assert args.num_encs == 1, "number of encoder should be 1 ({} is given)".format(
        args.num_encs
    )

    # load trained model parameters
    logging.info("reading model parameters from " + args.model)
    model_class = dynamic_import(train_args.model_module)
    model = model_class(idim, odim, train_args)
    assert isinstance(model, ASRInterface)
    torch_load(args.model, model)
    model.recog_args = args

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info("gpu id: " + str(gpu_id))
        model.cuda()

    # read json data
    with open(args.recog_json, "rb") as f:
        js = json.load(f)["utts"]

    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=False,
        sort_in_input_length=False,
        preprocess_conf=None,  # Apply pre_process in outer func
    )
    if args.batchsize == 0:
        args.batchsize = 1

    # Creates writers for outputs from the network
    if args.enh_wspecifier is not None:
        enh_writer = file_writer_helper(args.enh_wspecifier, filetype=args.enh_filetype)
    else:
        enh_writer = None

    # Creates a Transformation instance
    preprocess_conf = (
        train_args.preprocess_conf
        if args.preprocess_conf is None
        else args.preprocess_conf
    )
    if preprocess_conf is not None:
        logging.info(f"Use preprocessing: {preprocess_conf}")
        transform = Transformation(preprocess_conf)
    else:
        transform = None

    # Creates a IStft instance
    istft = None
    frame_shift = args.istft_n_shift  # Used for plot the spectrogram
    if args.apply_istft:
        if preprocess_conf is not None:
            # Read the conffile and find stft setting
            with open(preprocess_conf) as f:
                # Json format: e.g.
                #    {"process": [{"type": "stft",
                #                  "win_length": 400,
                #                  "n_fft": 512, "n_shift": 160,
                #                  "window": "han"},
                #                 {"type": "foo", ...}, ...]}
                conf = json.load(f)
                assert "process" in conf, conf
                # Find stft setting
                for p in conf["process"]:
                    if p["type"] == "stft":
                        istft = IStft(
                            win_length=p["win_length"],
                            n_shift=p["n_shift"],
                            window=p.get("window", "hann"),
                        )
                        logging.info(
                            "stft is found in {}. "
                            "Setting istft config from it\n{}".format(
                                preprocess_conf, istft
                            )
                        )
                        frame_shift = p["n_shift"]
                        break
        if istft is None:
            # Set from command line arguments
            istft = IStft(
                win_length=args.istft_win_length,
                n_shift=args.istft_n_shift,
                window=args.istft_window,
            )
            logging.info(
                "Setting istft config from the command line args\n{}".format(istft)
            )

    # sort data
    keys = list(js.keys())
    feat_lens = [js[key]["input"][0]["shape"][0] for key in keys]
    sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
    keys = [keys[i] for i in sorted_index]

    def grouper(n, iterable, fillvalue=None):
        kargs = [iter(iterable)] * n
        return itertools.zip_longest(*kargs, fillvalue=fillvalue)

    num_images = 0
    if not os.path.exists(args.image_dir):
        os.makedirs(args.image_dir)

    for names in grouper(args.batchsize, keys, None):
        batch = [(name, js[name]) for name in names]

        # May be in time region: (Batch, [Time, Channel])
        org_feats = load_inputs_and_targets(batch)[0]
        if transform is not None:
            # May be in time-freq region: : (Batch, [Time, Channel, Freq])
            feats = transform(org_feats, train=False)
        else:
            feats = org_feats

        with torch.no_grad():
            enhanced, mask, ilens = model.enhance(feats)

        for idx, name in enumerate(names):
            # Assuming mask, feats : [Batch, Time, Channel. Freq]
            #          enhanced    : [Batch, Time, Freq]
            enh = enhanced[idx][: ilens[idx]]
            mas = mask[idx][: ilens[idx]]
            feat = feats[idx]

            # Plot spectrogram
            if args.image_dir is not None and num_images < args.num_images:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                num_images += 1
                ref_ch = 0

                plt.figure(figsize=(20, 10))
                plt.subplot(4, 1, 1)
                plt.title("Mask [ref={}ch]".format(ref_ch))
                plot_spectrogram(
                    plt,
                    mas[:, ref_ch].T,
                    fs=args.fs,
                    mode="linear",
                    frame_shift=frame_shift,
                    bottom=False,
                    labelbottom=False,
                )

                plt.subplot(4, 1, 2)
                plt.title("Noisy speech [ref={}ch]".format(ref_ch))
                plot_spectrogram(
                    plt,
                    feat[:, ref_ch].T,
                    fs=args.fs,
                    mode="db",
                    frame_shift=frame_shift,
                    bottom=False,
                    labelbottom=False,
                )

                plt.subplot(4, 1, 3)
                plt.title("Masked speech [ref={}ch]".format(ref_ch))
                plot_spectrogram(
                    plt,
                    (feat[:, ref_ch] * mas[:, ref_ch]).T,
                    frame_shift=frame_shift,
                    fs=args.fs,
                    mode="db",
                    bottom=False,
                    labelbottom=False,
                )

                plt.subplot(4, 1, 4)
                plt.title("Enhanced speech")
                plot_spectrogram(
                    plt, enh.T, fs=args.fs, mode="db", frame_shift=frame_shift
                )

                plt.savefig(os.path.join(args.image_dir, name + ".png"))
                plt.clf()

            # Write enhanced wave files
            if enh_writer is not None:
                if istft is not None:
                    enh = istft(enh)
                else:
                    enh = enh

                if args.keep_length:
                    if len(org_feats[idx]) < len(enh):
                        # Truncate the frames added by stft padding
                        enh = enh[: len(org_feats[idx])]
                    elif len(org_feats) > len(enh):
                        padwidth = [(0, (len(org_feats[idx]) - len(enh)))] + [
                            (0, 0)
                        ] * (enh.ndim - 1)
                        enh = np.pad(enh, padwidth, mode="constant")

                if args.enh_filetype in ("sound", "sound.hdf5"):
                    enh_writer[name] = (args.fs, enh)
                else:
                    # Hint: To dump stft_signal, mask or etc,
                    # enh_filetype='hdf5' might be convenient.
                    enh_writer[name] = enh

            if num_images >= args.num_images and enh_writer is None:
                logging.info("Breaking the process.")
                break
