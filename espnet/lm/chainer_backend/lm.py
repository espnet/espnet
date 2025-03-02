#!/usr/bin/env python3
"""Chainer LM module."""

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This code is ported from the following implementation written in Torch.
# https://github.com/chainer/chainer/blob/master/examples/ptb/train_ptb_custom_loop.py


import copy
import json
import logging

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import link, reporter, training
from chainer.dataset import convert

# for classifier link
from chainer.functions.loss import softmax_cross_entropy
from chainer.training import extensions

import espnet.nets.chainer_backend.deterministic_embed_id as DL
from espnet.lm.lm_utils import (
    MakeSymlinkToBestModel,
    ParallelSentenceIterator,
    compute_perplexity,
    count_tokens,
    read_tokens,
)
from espnet.nets.lm_interface import LMInterface
from espnet.optimizer.factory import dynamic_import_optimizer
from espnet.scheduler.chainer import ChainerScheduler
from espnet.scheduler.scheduler import dynamic_import_scheduler
from espnet.utils.deterministic_utils import set_deterministic_chainer
from espnet.utils.training.evaluator import BaseEvaluator
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.tensorboard_logger import TensorboardLogger
from espnet.utils.training.train_utils import check_early_stop, set_early_stop


# TODO(karita): reimplement RNNLM with new interface
class DefaultRNNLM(LMInterface, link.Chain):
    """Default RNNLM wrapper to compute reduce framewise loss values.

    Args:
        n_vocab (int): The size of the vocabulary
        args (argparse.Namespace): configurations. see `add_arguments`
    """

    @staticmethod
    def add_arguments(parser):
        """Append additional arguments to parser."""
        parser.add_argument(
            "--type",
            type=str,
            default="lstm",
            nargs="?",
            choices=["lstm", "gru"],
            help="Which type of RNN to use",
        )
        parser.add_argument(
            "--layer", "-l", type=int, default=2, help="Number of hidden layers"
        )
        parser.add_argument(
            "--unit", "-u", type=int, default=650, help="Number of hidden units"
        )
        return parser


class ClassifierWithState(link.Chain):
    """A wrapper for a chainer RNNLM.

    :param link.Chain predictor : The RNNLM
    :param function lossfun: The loss function to use
    :param int/str label_key:
    """

    def __init__(
        self,
        predictor,
        lossfun=softmax_cross_entropy.softmax_cross_entropy,
        label_key=-1,
    ):
        """Initialize class."""
        if not (isinstance(label_key, (int, str))):
            raise TypeError("label_key must be int or str, but is %s" % type(label_key))

        super(ClassifierWithState, self).__init__()
        self.lossfun = lossfun
        self.y = None
        self.loss = None
        self.label_key = label_key

        with self.init_scope():
            self.predictor = predictor

    def __call__(self, state, *args, **kwargs):
        """Compute the loss value for an input and label pair.

            It also computes accuracy and stores it to the attribute.
            When ``label_key`` is ``int``, the corresponding element in ``args``
            is treated as ground truth labels. And when it is ``str``, the
            element in ``kwargs`` is used.
            The all elements of ``args`` and ``kwargs`` except the groundtruth
            labels are features.
            It feeds features to the predictor and compare the result
            with ground truth labels.

        :param state : The LM state
        :param list[chainer.Variable] args : Input minibatch
        :param dict[chainer.Variable] kwargs : Input minibatch
        :return loss value
        :rtype chainer.Variable
        """
        if isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = "Label key %d is out of bounds" % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[: self.label_key] + args[self.label_key + 1 :]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]

        self.y = None
        self.loss = None
        state, self.y = self.predictor(state, *args, **kwargs)
        self.loss = self.lossfun(self.y, t)
        return state, self.loss

    def predict(self, state, x):
        """Predict log probabilities for given state and input x using the predictor.

        :param state : the state
        :param x : the input
        :return a tuple (state, log prob vector)
        :rtype cupy/numpy array
        """
        if hasattr(self.predictor, "normalized") and self.predictor.normalized:
            return self.predictor(state, x)
        else:
            state, z = self.predictor(state, x)
            return state, F.log_softmax(z).data

    def final(self, state):
        """Predict final log probabilities for given state using the predictor.

        :param state : the state
        :return log probability vector
        :rtype cupy/numpy array

        """
        if hasattr(self.predictor, "final"):
            return self.predictor.final(state)
        else:
            return 0.0


# Definition of a recurrent net for language modeling
class RNNLM(chainer.Chain):
    """A chainer RNNLM.

    :param int n_vocab: The size of the vocabulary
    :param int n_layers: The number of layers to create
    :param int n_units: The number of units per layer
    :param str type: The RNN type
    """

    def __init__(self, n_vocab, n_layers, n_units, typ="lstm"):
        """Initialize class."""
        super(RNNLM, self).__init__()
        with self.init_scope():
            self.embed = DL.EmbedID(n_vocab, n_units)
            self.rnn = (
                chainer.ChainList(
                    *[L.StatelessLSTM(n_units, n_units) for _ in range(n_layers)]
                )
                if typ == "lstm"
                else chainer.ChainList(
                    *[L.StatelessGRU(n_units, n_units) for _ in range(n_layers)]
                )
            )
            self.lo = L.Linear(n_units, n_vocab)

        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        self.n_layers = n_layers
        self.n_units = n_units
        self.typ = typ

    def __call__(self, state, x):
        """Compute RNNLM call."""
        if state is None:
            if self.typ == "lstm":
                state = {"c": [None] * self.n_layers, "h": [None] * self.n_layers}
            else:
                state = {"h": [None] * self.n_layers}

        h = [None] * self.n_layers
        emb = self.embed(x)
        if self.typ == "lstm":
            c = [None] * self.n_layers
            c[0], h[0] = self.rnn[0](state["c"][0], state["h"][0], F.dropout(emb))
            for n in range(1, self.n_layers):
                c[n], h[n] = self.rnn[n](
                    state["c"][n], state["h"][n], F.dropout(h[n - 1])
                )
            state = {"c": c, "h": h}
        else:
            if state["h"][0] is None:
                xp = self.xp
                with chainer.backends.cuda.get_device_from_id(self._device_id):
                    state["h"][0] = chainer.Variable(
                        xp.zeros((emb.shape[0], self.n_units), dtype=emb.dtype)
                    )
            h[0] = self.rnn[0](state["h"][0], F.dropout(emb))
            for n in range(1, self.n_layers):
                if state["h"][n] is None:
                    xp = self.xp
                    with chainer.backends.cuda.get_device_from_id(self._device_id):
                        state["h"][n] = chainer.Variable(
                            xp.zeros(
                                (h[n - 1].shape[0], self.n_units), dtype=h[n - 1].dtype
                            )
                        )
                h[n] = self.rnn[n](state["h"][n], F.dropout(h[n - 1]))
            state = {"h": h}
        y = self.lo(F.dropout(h[-1]))
        return state, y


class BPTTUpdater(training.updaters.StandardUpdater):
    """An updater for a chainer LM.

    :param chainer.dataset.Iterator train_iter : The train iterator
    :param optimizer:
    :param schedulers:
    :param int device : The device id
    :param int accum_grad :
    """

    def __init__(self, train_iter, optimizer, schedulers, device, accum_grad):
        """Initialize BPTT Updater."""
        super(BPTTUpdater, self).__init__(train_iter, optimizer, device=device)
        self.scheduler = ChainerScheduler(schedulers, optimizer)
        self.accum_grad = accum_grad

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        """Run main update optimizer."""
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator("main")
        optimizer = self.get_optimizer("main")

        count = 0
        sum_loss = 0
        optimizer.target.cleargrads()  # Clear the parameter gradients
        for _ in range(self.accum_grad):
            # Progress the dataset iterator for sentences at each iteration.
            batch = train_iter.__next__()
            x, t = convert.concat_examples(batch, device=self.device, padding=(0, -1))
            # Concatenate the token IDs to matrices and send them to the device
            # self.converter does this job
            # (it is chainer.dataset.concat_examples by default)
            xp = chainer.backends.cuda.get_array_module(x)
            loss = 0
            state = None
            batch_size, sequence_length = x.shape
            for i in range(sequence_length):
                # Compute the loss at this time step and accumulate it
                state, loss_batch = optimizer.target(
                    state, chainer.Variable(x[:, i]), chainer.Variable(t[:, i])
                )
                non_zeros = xp.count_nonzero(x[:, i])
                loss += loss_batch * non_zeros
                count += int(non_zeros)
            # backward
            loss /= batch_size * self.accum_grad  # normalized by batch size
            sum_loss += float(loss.data)
            loss.backward()  # Backprop
            loss.unchain_backward()  # Truncate the graph

        reporter.report({"loss": sum_loss}, optimizer.target)
        reporter.report({"count": count}, optimizer.target)
        # update
        optimizer.update()  # Update the parameters
        self.scheduler.step(self.iteration)


class LMEvaluator(BaseEvaluator):
    """A custom evaluator for a chainer LM.

    :param chainer.dataset.Iterator val_iter : The validation iterator
    :param eval_model : The model to evaluate
    :param int device : The device id to use
    """

    def __init__(self, val_iter, eval_model, device):
        """Initialize LM Evaluator."""
        super(LMEvaluator, self).__init__(val_iter, eval_model, device=device)

    def evaluate(self):
        """Compute evaluation."""
        val_iter = self.get_iterator("main")
        target = self.get_target("main")
        loss = 0
        count = 0
        for batch in copy.copy(val_iter):
            x, t = convert.concat_examples(batch, device=self.device, padding=(0, -1))
            xp = chainer.backends.cuda.get_array_module(x)
            state = None
            for i in range(len(x[0])):
                state, loss_batch = target(state, x[:, i], t[:, i])
                non_zeros = xp.count_nonzero(x[:, i])
                loss += loss_batch.data * non_zeros
                count += int(non_zeros)
        # report validation loss
        observation = {}
        with reporter.report_scope(observation):
            reporter.report({"loss": float(loss / count)}, target)
        return observation


def train(args):
    """Train with the given args.

    :param Namespace args: The program arguments
    """
    # TODO(karita): support this
    if args.model_module != "default":
        raise NotImplementedError("chainer backend does not support --model-module")

    # display chainer version
    logging.info("chainer version = " + chainer.__version__)

    set_deterministic_chainer(args)

    # check cuda and cudnn availability
    if not chainer.cuda.available:
        logging.warning("cuda is not available")
    if not chainer.cuda.cudnn_enabled:
        logging.warning("cudnn is not available")

    # get special label ids
    unk = args.char_list_dict["<unk>"]
    eos = args.char_list_dict["<eos>"]
    # read tokens as a sequence of sentences
    train = read_tokens(args.train_label, args.char_list_dict)
    val = read_tokens(args.valid_label, args.char_list_dict)
    # count tokens
    n_train_tokens, n_train_oovs = count_tokens(train, unk)
    n_val_tokens, n_val_oovs = count_tokens(val, unk)
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
    train_iter = ParallelSentenceIterator(
        train,
        args.batchsize,
        max_length=args.maxlen,
        sos=eos,
        eos=eos,
        shuffle=not use_sortagrad,
    )
    val_iter = ParallelSentenceIterator(
        val, args.batchsize, max_length=args.maxlen, sos=eos, eos=eos, repeat=False
    )
    epoch_iters = int(len(train_iter.batch_indices) / args.accum_grad)
    logging.info("#iterations per epoch = %d" % epoch_iters)
    logging.info("#total iterations = " + str(args.epoch * epoch_iters))
    # Prepare an RNNLM model
    rnn = RNNLM(args.n_vocab, args.layer, args.unit, args.type)
    model = ClassifierWithState(rnn)
    if args.ngpu > 1:
        logging.warning("currently, multi-gpu is not supported. use single gpu.")
    if args.ngpu > 0:
        # Make the specified GPU current
        gpu_id = 0
        chainer.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()
    else:
        gpu_id = -1

    # Save model conf to json
    model_conf = args.outdir + "/model.json"
    with open(model_conf, "wb") as f:
        logging.info("writing a model config file to " + model_conf)
        f.write(
            json.dumps(vars(args), indent=4, ensure_ascii=False, sort_keys=True).encode(
                "utf_8"
            )
        )

    # Set up an optimizer
    opt_class = dynamic_import_optimizer(args.opt, args.backend)
    optimizer = opt_class.from_args(model, args)
    if args.schedulers is None:
        schedulers = []
    else:
        schedulers = [dynamic_import_scheduler(v)(k, args) for k, v in args.schedulers]

    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    updater = BPTTUpdater(train_iter, optimizer, schedulers, gpu_id, args.accum_grad)
    trainer = training.Trainer(updater, (args.epoch, "epoch"), out=args.outdir)
    trainer.extend(LMEvaluator(val_iter, model, device=gpu_id))
    trainer.extend(
        extensions.LogReport(
            postprocess=compute_perplexity,
            trigger=(args.report_interval_iters, "iteration"),
        )
    )
    trainer.extend(
        extensions.PrintReport(
            ["epoch", "iteration", "perplexity", "val_perplexity", "elapsed_time"]
        ),
        trigger=(args.report_interval_iters, "iteration"),
    )
    trainer.extend(extensions.ProgressBar(update_interval=args.report_interval_iters))
    trainer.extend(extensions.snapshot(filename="snapshot.ep.{.updater.epoch}"))
    trainer.extend(extensions.snapshot_object(model, "rnnlm.model.{.updater.epoch}"))
    # MEMO(Hori): wants to use MinValueTrigger, but it seems to fail in resuming
    trainer.extend(MakeSymlinkToBestModel("validation/main/loss", "rnnlm.model"))

    if use_sortagrad:
        trainer.extend(
            ShufflingEnabler([train_iter]),
            trigger=(args.sortagrad if args.sortagrad != -1 else args.epoch, "epoch"),
        )

    if args.resume:
        logging.info("resumed from %s" % args.resume)
        chainer.serializers.load_npz(args.resume, trainer)

    set_early_stop(trainer, args, is_lm=True)
    if args.tensorboard_dir is not None and args.tensorboard_dir != "":
        try:
            from tensorboardX import SummaryWriter
        except Exception:
            logging.error("Please install tensorboardx")
            raise
        writer = SummaryWriter(args.tensorboard_dir)
        trainer.extend(
            TensorboardLogger(writer), trigger=(args.report_interval_iters, "iteration")
        )

    trainer.run()
    check_early_stop(trainer, args.epoch)

    # compute perplexity for test set
    if args.test_label:
        logging.info("test the best model")
        chainer.serializers.load_npz(args.outdir + "/rnnlm.model.best", model)
        test = read_tokens(args.test_label, args.char_list_dict)
        n_test_tokens, n_test_oovs = count_tokens(test, unk)
        logging.info("#sentences in the test data = " + str(len(test)))
        logging.info("#tokens in the test data = " + str(n_test_tokens))
        logging.info(
            "oov rate in the test data = %.2f %%" % (n_test_oovs / n_test_tokens * 100)
        )
        test_iter = ParallelSentenceIterator(
            test, args.batchsize, max_length=args.maxlen, sos=eos, eos=eos, repeat=False
        )
        evaluator = LMEvaluator(test_iter, model, device=gpu_id)
        with chainer.using_config("train", False):
            result = evaluator()
        logging.info("test perplexity: " + str(np.exp(float(result["main/loss"]))))
