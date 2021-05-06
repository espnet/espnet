#!/usr/bin/env python3
# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# This code is ported from the following implementation written in Torch.
# https://github.com/chainer/chainer/blob/master/examples/ptb/train_ptb_custom_loop.py

"""LM training in pytorch."""

import copy
import json
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parallel import data_parallel

from chainer import Chain
from chainer.dataset import convert
from chainer import reporter
from chainer import training
from chainer.training import extensions

from espnet.lm.lm_utils import count_tokens
from espnet.lm.lm_utils import load_dataset
from espnet.lm.lm_utils import MakeSymlinkToBestModel
from espnet.lm.lm_utils import ParallelSentenceIterator
from espnet.lm.lm_utils import read_tokens
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.lm_interface import LMInterface
from espnet.optimizer.factory import dynamic_import_optimizer
from espnet.scheduler.pytorch import PyTorchScheduler
from espnet.scheduler.scheduler import dynamic_import_scheduler

from espnet.asr.asr_utils import snapshot_object
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_snapshot

from espnet.utils.training.tensorboard_logger import TensorboardLogger
from tensorboardX import SummaryWriter

from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.training.evaluator import BaseEvaluator
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop


def compute_perplexity(result):
    """Compute and add the perplexity to the LogReport.

    :param dict result: The current observations
    """
    # Routine to rewrite the result dictionary of LogReport to add perplexity values
    result["perplexity"] = np.exp(result["main/nll"] / result["main/count"])
    if "validation/main/nll" in result:
        result["val_perplexity"] = np.exp(
            result["validation/main/nll"] / result["validation/main/count"]
        )


class Reporter(Chain):
    """Dummy module to use chainer's trainer."""

    def report(self, loss):
        """Report nothing."""
        pass


def concat_examples(batch, device=None, padding=None):
    """Concat examples in minibatch.

    :param np.ndarray batch: The batch to concatenate
    :param int device: The device to send to
    :param Tuple[int,int] padding: The padding to use
    :return: (inputs, targets)
    :rtype (torch.Tensor, torch.Tensor)
    """
    x, t = convert.concat_examples(batch, padding=padding)
    x = torch.from_numpy(x)
    t = torch.from_numpy(t)
    if device is not None and device >= 0:
        x = x.cuda(device)
        t = t.cuda(device)
    return x, t


class BPTTUpdater(training.StandardUpdater):
    """An updater for a pytorch LM."""

    def __init__(
        self,
        train_iter,
        model,
        optimizer,
        schedulers,
        device,
        gradclip=None,
        use_apex=False,
        accum_grad=1,
    ):
        """Initialize class.

        Args:
            train_iter (chainer.dataset.Iterator): The train iterator
            model (LMInterface) : The model to update
            optimizer (torch.optim.Optimizer): The optimizer for training
            schedulers (espnet.scheduler.scheduler.SchedulerInterface):
                The schedulers of `optimizer`
            device (int): The device id
            gradclip (float): The gradient clipping value to use
            use_apex (bool): The flag to use Apex in backprop.
            accum_grad (int): The number of gradient accumulation.

        """
        super(BPTTUpdater, self).__init__(train_iter, optimizer)
        self.model = model
        self.device = device
        self.gradclip = gradclip
        self.use_apex = use_apex
        self.scheduler = PyTorchScheduler(schedulers, optimizer)
        self.accum_grad = accum_grad

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        """Update the model."""
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator("main")
        optimizer = self.get_optimizer("main")
        # Progress the dataset iterator for sentences at each iteration.
        self.model.zero_grad()  # Clear the parameter gradients
        accum = {"loss": 0.0, "nll": 0.0, "count": 0}
        for _ in range(self.accum_grad):
            batch = train_iter.__next__()
            # Concatenate the token IDs to matrices and send them to the device
            # self.converter does this job
            # (it is chainer.dataset.concat_examples by default)
            x, t = concat_examples(batch, device=self.device[0], padding=(0, -100))
            if self.device[0] == -1:
                loss, nll, count = self.model(x, t)
            else:
                # apex does not support torch.nn.DataParallel
                loss, nll, count = data_parallel(self.model, (x, t), self.device)

            # backward
            loss = loss.mean() / self.accum_grad
            if self.use_apex:
                from apex import amp

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()  # Backprop
            # accumulate stats
            accum["loss"] += float(loss)
            accum["nll"] += float(nll.sum())
            accum["count"] += int(count.sum())

        for k, v in accum.items():
            reporter.report({k: v}, optimizer.target)
        if self.gradclip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradclip)
        optimizer.step()  # Update the parameters
        self.scheduler.step(n_iter=self.iteration)


class LMEvaluator(BaseEvaluator):
    """A custom evaluator for a pytorch LM."""

    def __init__(self, val_iter, eval_model, reporter, device):
        """Initialize class.

        :param chainer.dataset.Iterator val_iter : The validation iterator
        :param LMInterface eval_model : The model to evaluate
        :param chainer.Reporter reporter : The observations reporter
        :param int device : The device id to use

        """
        super(LMEvaluator, self).__init__(val_iter, reporter, device=-1)
        self.model = eval_model
        self.device = device

    def evaluate(self):
        """Evaluate the model."""
        val_iter = self.get_iterator("main")
        loss = 0
        nll = 0
        count = 0
        self.model.eval()
        with torch.no_grad():
            for batch in copy.copy(val_iter):
                x, t = concat_examples(batch, device=self.device[0], padding=(0, -100))
                if self.device[0] == -1:
                    l, n, c = self.model(x, t)
                else:
                    # apex does not support torch.nn.DataParallel
                    l, n, c = data_parallel(self.model, (x, t), self.device)
                loss += float(l.sum())
                nll += float(n.sum())
                count += int(c.sum())
        self.model.train()
        # report validation loss
        observation = {}
        with reporter.report_scope(observation):
            reporter.report({"loss": loss}, self.model.reporter)
            reporter.report({"nll": nll}, self.model.reporter)
            reporter.report({"count": count}, self.model.reporter)
        return observation


def train(args):
    """Train with the given args.

    :param Namespace args: The program arguments
    :param type model_class: LMInterface class for training
    """
    model_class = dynamic_import_lm(args.model_module, args.backend)
    assert issubclass(model_class, LMInterface), "model should implement LMInterface"
    # display torch version
    logging.info("torch version = " + torch.__version__)

    set_deterministic_pytorch(args)

    # check cuda and cudnn availability
    if not torch.cuda.is_available():
        logging.warning("cuda is not available")

    # get special label ids
    unk = args.char_list_dict["<unk>"]
    eos = args.char_list_dict["<eos>"]
    # read tokens as a sequence of sentences
    val, n_val_tokens, n_val_oovs = load_dataset(
        args.valid_label, args.char_list_dict, args.dump_hdf5_path
    )
    train, n_train_tokens, n_train_oovs = load_dataset(
        args.train_label, args.char_list_dict, args.dump_hdf5_path
    )
    logging.info("#vocab = " + str(args.n_vocab))
    logging.info("#sentences in the training data = " + str(len(train)))
    logging.info("#tokens in the training data = " + str(n_train_tokens))
    logging.info(
        "oov rate in the training data = %.2f %%"
        % (n_train_oovs / n_train_tokens * 100)
    )
    logging.info("#sentences in the validation data = " + str(len(val)))
    logging.info("#tokens in the validation data = " + str(n_val_tokens))
    logging.info(
        "oov rate in the validation data = %.2f %%" % (n_val_oovs / n_val_tokens * 100)
    )

    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    # Create the dataset iterators
    batch_size = args.batchsize * max(args.ngpu, 1)
    if batch_size * args.accum_grad > args.batchsize:
        logging.info(
            f"batch size is automatically increased "
            f"({args.batchsize} -> {batch_size * args.accum_grad})"
        )
    train_iter = ParallelSentenceIterator(
        train,
        batch_size,
        max_length=args.maxlen,
        sos=eos,
        eos=eos,
        shuffle=not use_sortagrad,
    )
    val_iter = ParallelSentenceIterator(
        val, batch_size, max_length=args.maxlen, sos=eos, eos=eos, repeat=False
    )
    epoch_iters = int(len(train_iter.batch_indices) / args.accum_grad)
    logging.info("#iterations per epoch = %d" % epoch_iters)
    logging.info("#total iterations = " + str(args.epoch * epoch_iters))
    # Prepare an RNNLM model
    if args.train_dtype in ("float16", "float32", "float64"):
        dtype = getattr(torch, args.train_dtype)
    else:
        dtype = torch.float32
    model = model_class(args.n_vocab, args).to(dtype=dtype)
    if args.ngpu > 0:
        model.to("cuda")
        gpu_id = list(range(args.ngpu))
    else:
        gpu_id = [-1]

    # Save model conf to json
    model_conf = args.outdir + "/model.json"
    with open(model_conf, "wb") as f:
        logging.info("writing a model config file to " + model_conf)
        f.write(
            json.dumps(vars(args), indent=4, ensure_ascii=False, sort_keys=True).encode(
                "utf_8"
            )
        )

    logging.warning(
        "num. model params: {:,} (num. trained: {:,} ({:.1f}%))".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.parameters() if p.requires_grad)
            * 100.0
            / sum(p.numel() for p in model.parameters()),
        )
    )

    # Set up an optimizer
    opt_class = dynamic_import_optimizer(args.opt, args.backend)
    optimizer = opt_class.from_args(model.parameters(), args)
    if args.schedulers is None:
        schedulers = []
    else:
        schedulers = [dynamic_import_scheduler(v)(k, args) for k, v in args.schedulers]

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
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.train_dtype)
        use_apex = True
    else:
        use_apex = False

    # FIXME: TOO DIRTY HACK
    reporter = Reporter()
    setattr(model, "reporter", reporter)
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    updater = BPTTUpdater(
        train_iter,
        model,
        optimizer,
        schedulers,
        gpu_id,
        gradclip=args.gradclip,
        use_apex=use_apex,
        accum_grad=args.accum_grad,
    )
    trainer = training.Trainer(updater, (args.epoch, "epoch"), out=args.outdir)
    trainer.extend(LMEvaluator(val_iter, model, reporter, device=gpu_id))
    trainer.extend(
        extensions.LogReport(
            postprocess=compute_perplexity,
            trigger=(args.report_interval_iters, "iteration"),
        )
    )
    trainer.extend(
        extensions.PrintReport(
            [
                "epoch",
                "iteration",
                "main/loss",
                "perplexity",
                "val_perplexity",
                "elapsed_time",
            ]
        ),
        trigger=(args.report_interval_iters, "iteration"),
    )
    trainer.extend(extensions.ProgressBar(update_interval=args.report_interval_iters))
    # Save best models
    trainer.extend(torch_snapshot(filename="snapshot.ep.{.updater.epoch}"))
    trainer.extend(snapshot_object(model, "rnnlm.model.{.updater.epoch}"))
    # T.Hori: MinValueTrigger should be used, but it fails when resuming
    trainer.extend(MakeSymlinkToBestModel("validation/main/loss", "rnnlm.model"))

    if use_sortagrad:
        trainer.extend(
            ShufflingEnabler([train_iter]),
            trigger=(args.sortagrad if args.sortagrad != -1 else args.epoch, "epoch"),
        )
    if args.resume:
        logging.info("resumed from %s" % args.resume)
        torch_resume(args.resume, trainer)

    set_early_stop(trainer, args, is_lm=True)
    if args.tensorboard_dir is not None and args.tensorboard_dir != "":
        writer = SummaryWriter(args.tensorboard_dir)
        trainer.extend(
            TensorboardLogger(writer), trigger=(args.report_interval_iters, "iteration")
        )

    trainer.run()
    check_early_stop(trainer, args.epoch)

    # compute perplexity for test set
    if args.test_label:
        logging.info("test the best model")
        torch_load(args.outdir + "/rnnlm.model.best", model)
        test = read_tokens(args.test_label, args.char_list_dict)
        n_test_tokens, n_test_oovs = count_tokens(test, unk)
        logging.info("#sentences in the test data = " + str(len(test)))
        logging.info("#tokens in the test data = " + str(n_test_tokens))
        logging.info(
            "oov rate in the test data = %.2f %%" % (n_test_oovs / n_test_tokens * 100)
        )
        test_iter = ParallelSentenceIterator(
            test, batch_size, max_length=args.maxlen, sos=eos, eos=eos, repeat=False
        )
        evaluator = LMEvaluator(test_iter, model, reporter, device=gpu_id)
        result = evaluator()
        compute_perplexity(result)
        logging.info(f"test perplexity: {result['perplexity']}")
