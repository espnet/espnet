# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import copy
import json
import logging
import os
import shutil
import tempfile

import numpy as np
import torch

# * -------------------- training iterator related -------------------- *


class CompareValueTrigger(object):
    """Trigger invoked when key value getting bigger or lower than before.

    Args:
        key (str) : Key of value.
        compare_fn ((float, float) -> bool) : Function to compare the values.
        trigger (tuple(int, str)) : Trigger that decide the comparison interval.

    """

    def __init__(self, key, compare_fn, trigger=(1, "epoch")):
        from chainer import training

        self._key = key
        self._best_value = None
        self._interval_trigger = training.util.get_trigger(trigger)
        self._init_summary()
        self._compare_fn = compare_fn

    def __call__(self, trainer):
        """Get value related to the key and compare with current value."""
        observation = trainer.observation
        summary = self._summary
        key = self._key
        if key in observation:
            summary.add({key: observation[key]})

        if not self._interval_trigger(trainer):
            return False

        stats = summary.compute_mean()
        value = float(stats[key])  # copy to CPU
        self._init_summary()

        if self._best_value is None:
            # initialize best value
            self._best_value = value
            return False
        elif self._compare_fn(self._best_value, value):
            return True
        else:
            self._best_value = value
            return False

    def _init_summary(self):
        import chainer

        self._summary = chainer.reporter.DictSummary()


try:
    from chainer.training import extension
except ImportError:
    PlotAttentionReport = None
else:

    class PlotAttentionReport(extension.Extension):
        """Plot attention reporter.

        Args:
            att_vis_fn (espnet.nets.*_backend.e2e_asr.E2E.calculate_all_attentions):
                Function of attention visualization.
            data (list[tuple(str, dict[str, list[Any]])]): List json utt key items.
            outdir (str): Directory to save figures.
            converter (espnet.asr.*_backend.asr.CustomConverter):
                Function to convert data.
            device (int | torch.device): Device.
            reverse (bool): If True, input and output length are reversed.
            ikey (str): Key to access input
                (for ASR/ST ikey="input", for MT ikey="output".)
            iaxis (int): Dimension to access input
                (for ASR/ST iaxis=0, for MT iaxis=1.)
            okey (str): Key to access output
                (for ASR/ST okey="input", MT okay="output".)
            oaxis (int): Dimension to access output
                (for ASR/ST oaxis=0, for MT oaxis=0.)
            subsampling_factor (int): subsampling factor in encoder

        """

        def __init__(
            self,
            att_vis_fn,
            data,
            outdir,
            converter,
            transform,
            device,
            reverse=False,
            ikey="input",
            iaxis=0,
            okey="output",
            oaxis=0,
            subsampling_factor=1,
        ):
            self.att_vis_fn = att_vis_fn
            self.data = copy.deepcopy(data)
            self.data_dict = {k: v for k, v in copy.deepcopy(data)}
            # key is utterance ID
            self.outdir = outdir
            self.converter = converter
            self.transform = transform
            self.device = device
            self.reverse = reverse
            self.ikey = ikey
            self.iaxis = iaxis
            self.okey = okey
            self.oaxis = oaxis
            self.factor = subsampling_factor
            if not os.path.exists(self.outdir):
                os.makedirs(self.outdir)

        def __call__(self, trainer):
            """Plot and save image file of att_ws matrix."""
            att_ws, uttid_list = self.get_attention_weights()
            if isinstance(att_ws, list):  # multi-encoder case
                num_encs = len(att_ws) - 1
                # atts
                for i in range(num_encs):
                    for idx, att_w in enumerate(att_ws[i]):
                        filename = "%s/%s.ep.{.updater.epoch}.att%d.png" % (
                            self.outdir,
                            uttid_list[idx],
                            i + 1,
                        )
                        att_w = self.trim_attention_weight(uttid_list[idx], att_w)
                        np_filename = "%s/%s.ep.{.updater.epoch}.att%d.npy" % (
                            self.outdir,
                            uttid_list[idx],
                            i + 1,
                        )
                        np.save(np_filename.format(trainer), att_w)
                        self._plot_and_save_attention(att_w, filename.format(trainer))
                # han
                for idx, att_w in enumerate(att_ws[num_encs]):
                    filename = "%s/%s.ep.{.updater.epoch}.han.png" % (
                        self.outdir,
                        uttid_list[idx],
                    )
                    att_w = self.trim_attention_weight(uttid_list[idx], att_w)
                    np_filename = "%s/%s.ep.{.updater.epoch}.han.npy" % (
                        self.outdir,
                        uttid_list[idx],
                    )
                    np.save(np_filename.format(trainer), att_w)
                    self._plot_and_save_attention(
                        att_w, filename.format(trainer), han_mode=True
                    )
            else:
                for idx, att_w in enumerate(att_ws):
                    filename = "%s/%s.ep.{.updater.epoch}.png" % (
                        self.outdir,
                        uttid_list[idx],
                    )
                    att_w = self.trim_attention_weight(uttid_list[idx], att_w)
                    np_filename = "%s/%s.ep.{.updater.epoch}.npy" % (
                        self.outdir,
                        uttid_list[idx],
                    )
                    np.save(np_filename.format(trainer), att_w)
                    self._plot_and_save_attention(att_w, filename.format(trainer))

        def log_attentions(self, logger, step):
            """Add image files of att_ws matrix to the tensorboard."""
            att_ws, uttid_list = self.get_attention_weights()
            if isinstance(att_ws, list):  # multi-encoder case
                num_encs = len(att_ws) - 1
                # atts
                for i in range(num_encs):
                    for idx, att_w in enumerate(att_ws[i]):
                        att_w = self.trim_attention_weight(uttid_list[idx], att_w)
                        plot = self.draw_attention_plot(att_w)
                        logger.add_figure(
                            "%s_att%d" % (uttid_list[idx], i + 1), plot.gcf(), step,
                        )
                # han
                for idx, att_w in enumerate(att_ws[num_encs]):
                    att_w = self.trim_attention_weight(uttid_list[idx], att_w)
                    plot = self.draw_han_plot(att_w)
                    logger.add_figure(
                        "%s_han" % (uttid_list[idx]), plot.gcf(), step,
                    )
            else:
                for idx, att_w in enumerate(att_ws):
                    att_w = self.trim_attention_weight(uttid_list[idx], att_w)
                    plot = self.draw_attention_plot(att_w)
                    logger.add_figure("%s" % (uttid_list[idx]), plot.gcf(), step)

        def get_attention_weights(self):
            """Return attention weights.

            Returns:
                numpy.ndarray: attention weights. float. Its shape would be
                    differ from backend.
                    * pytorch-> 1) multi-head case => (B, H, Lmax, Tmax), 2)
                      other case => (B, Lmax, Tmax).
                    * chainer-> (B, Lmax, Tmax)

            """
            return_batch, uttid_list = self.transform(self.data, return_uttid=True)
            batch = self.converter([return_batch], self.device)
            if isinstance(batch, tuple):
                att_ws = self.att_vis_fn(*batch)
            else:
                att_ws = self.att_vis_fn(**batch)
            return att_ws, uttid_list

        def trim_attention_weight(self, uttid, att_w):
            """Transform attention matrix with regard to self.reverse."""
            if self.reverse:
                enc_key, enc_axis = self.okey, self.oaxis
                dec_key, dec_axis = self.ikey, self.iaxis
            else:
                enc_key, enc_axis = self.ikey, self.iaxis
                dec_key, dec_axis = self.okey, self.oaxis
            dec_len = int(self.data_dict[uttid][dec_key][dec_axis]["shape"][0])
            enc_len = int(self.data_dict[uttid][enc_key][enc_axis]["shape"][0])
            if self.factor > 1:
                enc_len //= self.factor
            if len(att_w.shape) == 3:
                att_w = att_w[:, :dec_len, :enc_len]
            else:
                att_w = att_w[:dec_len, :enc_len]
            return att_w

        def draw_attention_plot(self, att_w):
            """Plot the att_w matrix.

            Returns:
                matplotlib.pyplot: pyplot object with attention matrix image.

            """
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.clf()
            att_w = att_w.astype(np.float32)
            if len(att_w.shape) == 3:
                for h, aw in enumerate(att_w, 1):
                    plt.subplot(1, len(att_w), h)
                    plt.imshow(aw, aspect="auto")
                    plt.xlabel("Encoder Index")
                    plt.ylabel("Decoder Index")
            else:
                plt.imshow(att_w, aspect="auto")
                plt.xlabel("Encoder Index")
                plt.ylabel("Decoder Index")
            plt.tight_layout()
            return plt

        def draw_han_plot(self, att_w):
            """Plot the att_w matrix for hierarchical attention.

            Returns:
                matplotlib.pyplot: pyplot object with attention matrix image.

            """
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.clf()
            if len(att_w.shape) == 3:
                for h, aw in enumerate(att_w, 1):
                    legends = []
                    plt.subplot(1, len(att_w), h)
                    for i in range(aw.shape[1]):
                        plt.plot(aw[:, i])
                        legends.append("Att{}".format(i))
                    plt.ylim([0, 1.0])
                    plt.xlim([0, aw.shape[0]])
                    plt.grid(True)
                    plt.ylabel("Attention Weight")
                    plt.xlabel("Decoder Index")
                    plt.legend(legends)
            else:
                legends = []
                for i in range(att_w.shape[1]):
                    plt.plot(att_w[:, i])
                    legends.append("Att{}".format(i))
                plt.ylim([0, 1.0])
                plt.xlim([0, att_w.shape[0]])
                plt.grid(True)
                plt.ylabel("Attention Weight")
                plt.xlabel("Decoder Index")
                plt.legend(legends)
            plt.tight_layout()
            return plt

        def _plot_and_save_attention(self, att_w, filename, han_mode=False):
            if han_mode:
                plt = self.draw_han_plot(att_w)
            else:
                plt = self.draw_attention_plot(att_w)
            plt.savefig(filename)
            plt.close()


try:
    from chainer.training import extension
except ImportError:
    PlotCTCReport = None
else:

    class PlotCTCReport(extension.Extension):
        """Plot CTC reporter.

        Args:
            ctc_vis_fn (espnet.nets.*_backend.e2e_asr.E2E.calculate_all_ctc_probs):
                Function of CTC visualization.
            data (list[tuple(str, dict[str, list[Any]])]): List json utt key items.
            outdir (str): Directory to save figures.
            converter (espnet.asr.*_backend.asr.CustomConverter):
                Function to convert data.
            device (int | torch.device): Device.
            reverse (bool): If True, input and output length are reversed.
            ikey (str): Key to access input
                (for ASR/ST ikey="input", for MT ikey="output".)
            iaxis (int): Dimension to access input
                (for ASR/ST iaxis=0, for MT iaxis=1.)
            okey (str): Key to access output
                (for ASR/ST okey="input", MT okay="output".)
            oaxis (int): Dimension to access output
                (for ASR/ST oaxis=0, for MT oaxis=0.)
            subsampling_factor (int): subsampling factor in encoder

        """

        def __init__(
            self,
            ctc_vis_fn,
            data,
            outdir,
            converter,
            transform,
            device,
            reverse=False,
            ikey="input",
            iaxis=0,
            okey="output",
            oaxis=0,
            subsampling_factor=1,
        ):
            self.ctc_vis_fn = ctc_vis_fn
            self.data = copy.deepcopy(data)
            self.data_dict = {k: v for k, v in copy.deepcopy(data)}
            # key is utterance ID
            self.outdir = outdir
            self.converter = converter
            self.transform = transform
            self.device = device
            self.reverse = reverse
            self.ikey = ikey
            self.iaxis = iaxis
            self.okey = okey
            self.oaxis = oaxis
            self.factor = subsampling_factor
            if not os.path.exists(self.outdir):
                os.makedirs(self.outdir)

        def __call__(self, trainer):
            """Plot and save image file of ctc prob."""
            ctc_probs, uttid_list = self.get_ctc_probs()
            if isinstance(ctc_probs, list):  # multi-encoder case
                num_encs = len(ctc_probs) - 1
                for i in range(num_encs):
                    for idx, ctc_prob in enumerate(ctc_probs[i]):
                        filename = "%s/%s.ep.{.updater.epoch}.ctc%d.png" % (
                            self.outdir,
                            uttid_list[idx],
                            i + 1,
                        )
                        ctc_prob = self.trim_ctc_prob(uttid_list[idx], ctc_prob)
                        np_filename = "%s/%s.ep.{.updater.epoch}.ctc%d.npy" % (
                            self.outdir,
                            uttid_list[idx],
                            i + 1,
                        )
                        np.save(np_filename.format(trainer), ctc_prob)
                        self._plot_and_save_ctc(ctc_prob, filename.format(trainer))
            else:
                for idx, ctc_prob in enumerate(ctc_probs):
                    filename = "%s/%s.ep.{.updater.epoch}.png" % (
                        self.outdir,
                        uttid_list[idx],
                    )
                    ctc_prob = self.trim_ctc_prob(uttid_list[idx], ctc_prob)
                    np_filename = "%s/%s.ep.{.updater.epoch}.npy" % (
                        self.outdir,
                        uttid_list[idx],
                    )
                    np.save(np_filename.format(trainer), ctc_prob)
                    self._plot_and_save_ctc(ctc_prob, filename.format(trainer))

        def log_ctc_probs(self, logger, step):
            """Add image files of ctc probs to the tensorboard."""
            ctc_probs, uttid_list = self.get_ctc_probs()
            if isinstance(ctc_probs, list):  # multi-encoder case
                num_encs = len(ctc_probs) - 1
                for i in range(num_encs):
                    for idx, ctc_prob in enumerate(ctc_probs[i]):
                        ctc_prob = self.trim_ctc_prob(uttid_list[idx], ctc_prob)
                        plot = self.draw_ctc_plot(ctc_prob)
                        logger.add_figure(
                            "%s_ctc%d" % (uttid_list[idx], i + 1), plot.gcf(), step,
                        )
            else:
                for idx, ctc_prob in enumerate(ctc_probs):
                    ctc_prob = self.trim_ctc_prob(uttid_list[idx], ctc_prob)
                    plot = self.draw_ctc_plot(ctc_prob)
                    logger.add_figure("%s" % (uttid_list[idx]), plot.gcf(), step)

        def get_ctc_probs(self):
            """Return CTC probs.

            Returns:
                numpy.ndarray: CTC probs. float. Its shape would be
                    differ from backend. (B, Tmax, vocab).

            """
            return_batch, uttid_list = self.transform(self.data, return_uttid=True)
            batch = self.converter([return_batch], self.device)
            if isinstance(batch, tuple):
                probs = self.ctc_vis_fn(*batch)
            else:
                probs = self.ctc_vis_fn(**batch)
            return probs, uttid_list

        def trim_ctc_prob(self, uttid, prob):
            """Trim CTC posteriors accoding to input lengths."""
            enc_len = int(self.data_dict[uttid][self.ikey][self.iaxis]["shape"][0])
            if self.factor > 1:
                enc_len //= self.factor
            prob = prob[:enc_len]
            return prob

        def draw_ctc_plot(self, ctc_prob):
            """Plot the ctc_prob matrix.

            Returns:
                matplotlib.pyplot: pyplot object with CTC prob matrix image.

            """
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            ctc_prob = ctc_prob.astype(np.float32)

            plt.clf()
            topk_ids = np.argsort(ctc_prob, axis=1)
            n_frames, vocab = ctc_prob.shape
            times_probs = np.arange(n_frames)

            plt.figure(figsize=(20, 8))

            # NOTE: index 0 is reserved for blank
            for idx in set(topk_ids.reshape(-1).tolist()):
                if idx == 0:
                    plt.plot(
                        times_probs, ctc_prob[:, 0], ":", label="<blank>", color="grey"
                    )
                else:
                    plt.plot(times_probs, ctc_prob[:, idx])
            plt.xlabel("Input [frame]", fontsize=12)
            plt.ylabel("Posteriors", fontsize=12)
            plt.xticks(list(range(0, int(n_frames) + 1, 10)))
            plt.yticks(list(range(0, 2, 1)))
            plt.tight_layout()
            return plt

        def _plot_and_save_ctc(self, ctc_prob, filename):
            plt = self.draw_ctc_plot(ctc_prob)
            plt.savefig(filename)
            plt.close()


def restore_snapshot(model, snapshot, load_fn=None):
    """Extension to restore snapshot.

    Returns:
        An extension function.

    """
    import chainer
    from chainer import training

    if load_fn is None:
        load_fn = chainer.serializers.load_npz

    @training.make_extension(trigger=(1, "epoch"))
    def restore_snapshot(trainer):
        _restore_snapshot(model, snapshot, load_fn)

    return restore_snapshot


def _restore_snapshot(model, snapshot, load_fn=None):
    if load_fn is None:
        import chainer

        load_fn = chainer.serializers.load_npz

    load_fn(snapshot, model)
    logging.info("restored from " + str(snapshot))


def adadelta_eps_decay(eps_decay):
    """Extension to perform adadelta eps decay.

    Args:
        eps_decay (float): Decay rate of eps.

    Returns:
        An extension function.

    """
    from chainer import training

    @training.make_extension(trigger=(1, "epoch"))
    def adadelta_eps_decay(trainer):
        _adadelta_eps_decay(trainer, eps_decay)

    return adadelta_eps_decay


def _adadelta_eps_decay(trainer, eps_decay):
    optimizer = trainer.updater.get_optimizer("main")
    # for chainer
    if hasattr(optimizer, "eps"):
        current_eps = optimizer.eps
        setattr(optimizer, "eps", current_eps * eps_decay)
        logging.info("adadelta eps decayed to " + str(optimizer.eps))
    # pytorch
    else:
        for p in optimizer.param_groups:
            p["eps"] *= eps_decay
            logging.info("adadelta eps decayed to " + str(p["eps"]))


def adam_lr_decay(eps_decay):
    """Extension to perform adam lr decay.

    Args:
        eps_decay (float): Decay rate of lr.

    Returns:
        An extension function.

    """
    from chainer import training

    @training.make_extension(trigger=(1, "epoch"))
    def adam_lr_decay(trainer):
        _adam_lr_decay(trainer, eps_decay)

    return adam_lr_decay


def _adam_lr_decay(trainer, eps_decay):
    optimizer = trainer.updater.get_optimizer("main")
    # for chainer
    if hasattr(optimizer, "lr"):
        current_lr = optimizer.lr
        setattr(optimizer, "lr", current_lr * eps_decay)
        logging.info("adam lr decayed to " + str(optimizer.lr))
    # pytorch
    else:
        for p in optimizer.param_groups:
            p["lr"] *= eps_decay
            logging.info("adam lr decayed to " + str(p["lr"]))


def torch_snapshot(savefun=torch.save, filename="snapshot.ep.{.updater.epoch}"):
    """Extension to take snapshot of the trainer for pytorch.

    Returns:
        An extension function.

    """
    from chainer.training import extension

    @extension.make_extension(trigger=(1, "epoch"), priority=-100)
    def torch_snapshot(trainer):
        _torch_snapshot_object(trainer, trainer, filename.format(trainer), savefun)

    return torch_snapshot


def _torch_snapshot_object(trainer, target, filename, savefun):
    from chainer.serializers import DictionarySerializer

    # make snapshot_dict dictionary
    s = DictionarySerializer()
    s.save(trainer)
    if hasattr(trainer.updater.model, "model"):
        # (for TTS)
        if hasattr(trainer.updater.model.model, "module"):
            model_state_dict = trainer.updater.model.model.module.state_dict()
        else:
            model_state_dict = trainer.updater.model.model.state_dict()
    else:
        # (for ASR)
        if hasattr(trainer.updater.model, "module"):
            model_state_dict = trainer.updater.model.module.state_dict()
        else:
            model_state_dict = trainer.updater.model.state_dict()
    snapshot_dict = {
        "trainer": s.target,
        "model": model_state_dict,
        "optimizer": trainer.updater.get_optimizer("main").state_dict(),
    }

    # save snapshot dictionary
    fn = filename.format(trainer)
    prefix = "tmp" + fn
    tmpdir = tempfile.mkdtemp(prefix=prefix, dir=trainer.out)
    tmppath = os.path.join(tmpdir, fn)
    try:
        savefun(snapshot_dict, tmppath)
        shutil.move(tmppath, os.path.join(trainer.out, fn))
    finally:
        shutil.rmtree(tmpdir)


def add_gradient_noise(model, iteration, duration=100, eta=1.0, scale_factor=0.55):
    """Adds noise from a standard normal distribution to the gradients.

    The standard deviation (`sigma`) is controlled by the three hyper-parameters below.
    `sigma` goes to zero (no noise) with more iterations.

    Args:
        model (torch.nn.model): Model.
        iteration (int): Number of iterations.
        duration (int) {100, 1000}:
            Number of durations to control the interval of the `sigma` change.
        eta (float) {0.01, 0.3, 1.0}: The magnitude of `sigma`.
        scale_factor (float) {0.55}: The scale of `sigma`.
    """
    interval = (iteration // duration) + 1
    sigma = eta / interval ** scale_factor
    for param in model.parameters():
        if param.grad is not None:
            _shape = param.grad.size()
            noise = sigma * torch.randn(_shape).to(param.device)
            param.grad += noise


# * -------------------- general -------------------- *
def get_model_conf(model_path, conf_path=None):
    """Get model config information by reading a model config file (model.json).

    Args:
        model_path (str): Model path.
        conf_path (str): Optional model config path.

    Returns:
        list[int, int, dict[str, Any]]: Config information loaded from json file.

    """
    if conf_path is None:
        model_conf = os.path.dirname(model_path) + "/model.json"
    else:
        model_conf = conf_path
    with open(model_conf, "rb") as f:
        logging.info("reading a config file from " + model_conf)
        confs = json.load(f)
    if isinstance(confs, dict):
        # for lm
        args = confs
        return argparse.Namespace(**args)
    else:
        # for asr, tts, mt
        idim, odim, args = confs
        return idim, odim, argparse.Namespace(**args)


def chainer_load(path, model):
    """Load chainer model parameters.

    Args:
        path (str): Model path or snapshot file path to be loaded.
        model (chainer.Chain): Chainer model.

    """
    import chainer

    if "snapshot" in os.path.basename(path):
        chainer.serializers.load_npz(path, model, path="updater/model:main/")
    else:
        chainer.serializers.load_npz(path, model)


def torch_save(path, model):
    """Save torch model states.

    Args:
        path (str): Model path to be saved.
        model (torch.nn.Module): Torch model.

    """
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def snapshot_object(target, filename):
    """Returns a trainer extension to take snapshots of a given object.

    Args:
        target (model): Object to serialize.
        filename (str): Name of the file into which the object is serialized.It can
            be a format string, where the trainer object is passed to
            the :meth: `str.format` method. For example,
            ``'snapshot_{.updater.iteration}'`` is converted to
            ``'snapshot_10000'`` at the 10,000th iteration.

    Returns:
        An extension function.

    """
    from chainer.training import extension

    @extension.make_extension(trigger=(1, "epoch"), priority=-100)
    def snapshot_object(trainer):
        torch_save(os.path.join(trainer.out, filename.format(trainer)), target)

    return snapshot_object


def torch_load(path, model):
    """Load torch model states.

    Args:
        path (str): Model path or snapshot file path to be loaded.
        model (torch.nn.Module): Torch model.

    """
    if "snapshot" in os.path.basename(path):
        model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)[
            "model"
        ]
    else:
        model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)

    if hasattr(model, "module"):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)

    del model_state_dict


def torch_resume(snapshot_path, trainer):
    """Resume from snapshot for pytorch.

    Args:
        snapshot_path (str): Snapshot file path.
        trainer (chainer.training.Trainer): Chainer's trainer instance.

    """
    from chainer.serializers import NpzDeserializer

    # load snapshot
    snapshot_dict = torch.load(snapshot_path, map_location=lambda storage, loc: storage)

    # restore trainer states
    d = NpzDeserializer(snapshot_dict["trainer"])
    d.load(trainer)

    # restore model states
    if hasattr(trainer.updater.model, "model"):
        # (for TTS model)
        if hasattr(trainer.updater.model.model, "module"):
            trainer.updater.model.model.module.load_state_dict(snapshot_dict["model"])
        else:
            trainer.updater.model.model.load_state_dict(snapshot_dict["model"])
    else:
        # (for ASR model)
        if hasattr(trainer.updater.model, "module"):
            trainer.updater.model.module.load_state_dict(snapshot_dict["model"])
        else:
            trainer.updater.model.load_state_dict(snapshot_dict["model"])

    # retore optimizer states
    trainer.updater.get_optimizer("main").load_state_dict(snapshot_dict["optimizer"])

    # delete opened snapshot
    del snapshot_dict


# * ------------------ recognition related ------------------ *
def parse_hypothesis(hyp, char_list):
    """Parse hypothesis.

    Args:
        hyp (list[dict[str, Any]]): Recognition hypothesis.
        char_list (list[str]): List of characters.

    Returns:
        tuple(str, str, str, float)

    """
    # remove sos and get results
    tokenid_as_list = list(map(int, hyp["yseq"][1:]))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    score = float(hyp["score"])

    # convert to string
    tokenid = " ".join([str(idx) for idx in tokenid_as_list])
    token = " ".join(token_as_list)
    text = "".join(token_as_list).replace("<space>", " ")

    return text, token, tokenid, score


def add_results_to_json(js, nbest_hyps, char_list):
    """Add N-best results to json.

    Args:
        js (dict[str, Any]): Groundtruth utterance dict.
        nbest_hyps_sd (list[dict[str, Any]]):
            List of hypothesis for multi_speakers: nutts x nspkrs.
        char_list (list[str]): List of characters.

    Returns:
        dict[str, Any]: N-best results added utterance dict.

    """
    # copy old json info
    new_js = dict()
    new_js["utt2spk"] = js["utt2spk"]
    new_js["output"] = []

    for n, hyp in enumerate(nbest_hyps, 1):
        # parse hypothesis
        rec_text, rec_token, rec_tokenid, score = parse_hypothesis(hyp, char_list)

        # copy ground-truth
        if len(js["output"]) > 0:
            out_dic = dict(js["output"][0].items())
        else:
            # for no reference case (e.g., speech translation)
            out_dic = {"name": ""}

        # update name
        out_dic["name"] += "[%d]" % n

        # add recognition results
        out_dic["rec_text"] = rec_text
        out_dic["rec_token"] = rec_token
        out_dic["rec_tokenid"] = rec_tokenid
        out_dic["score"] = score

        # add to list of N-best result dicts
        new_js["output"].append(out_dic)

        # show 1-best result
        if n == 1:
            if "text" in out_dic.keys():
                logging.info("groundtruth: %s" % out_dic["text"])
            logging.info("prediction : %s" % out_dic["rec_text"])

    return new_js


def plot_spectrogram(
    plt,
    spec,
    mode="db",
    fs=None,
    frame_shift=None,
    bottom=True,
    left=True,
    right=True,
    top=False,
    labelbottom=True,
    labelleft=True,
    labelright=True,
    labeltop=False,
    cmap="inferno",
):
    """Plot spectrogram using matplotlib.

    Args:
        plt (matplotlib.pyplot): pyplot object.
        spec (numpy.ndarray): Input stft (Freq, Time)
        mode (str): db or linear.
        fs (int): Sample frequency. To convert y-axis to kHz unit.
        frame_shift (int): The frame shift of stft. To convert x-axis to second unit.
        bottom (bool):Whether to draw the respective ticks.
        left (bool):
        right (bool):
        top (bool):
        labelbottom (bool):Whether to draw the respective tick labels.
        labelleft (bool):
        labelright (bool):
        labeltop (bool):
        cmap (str): Colormap defined in matplotlib.

    """
    spec = np.abs(spec)
    if mode == "db":
        x = 20 * np.log10(spec + np.finfo(spec.dtype).eps)
    elif mode == "linear":
        x = spec
    else:
        raise ValueError(mode)

    if fs is not None:
        ytop = fs / 2000
        ylabel = "kHz"
    else:
        ytop = x.shape[0]
        ylabel = "bin"

    if frame_shift is not None and fs is not None:
        xtop = x.shape[1] * frame_shift / fs
        xlabel = "s"
    else:
        xtop = x.shape[1]
        xlabel = "frame"

    extent = (0, xtop, 0, ytop)
    plt.imshow(x[::-1], cmap=cmap, extent=extent)

    if labelbottom:
        plt.xlabel("time [{}]".format(xlabel))
    if labelleft:
        plt.ylabel("freq [{}]".format(ylabel))
    plt.colorbar().set_label("{}".format(mode))

    plt.tick_params(
        bottom=bottom,
        left=left,
        right=right,
        top=top,
        labelbottom=labelbottom,
        labelleft=labelleft,
        labelright=labelright,
        labeltop=labeltop,
    )
    plt.axis("auto")


# * ------------------ recognition related ------------------ *
def format_mulenc_args(args):
    """Format args for multi-encoder setup.

    It deals with following situations:  (when args.num_encs=2):
    1. args.elayers = None -> args.elayers = [4, 4];
    2. args.elayers = 4 -> args.elayers = [4, 4];
    3. args.elayers = [4, 4, 4] -> args.elayers = [4, 4].

    """
    # default values when None is assigned.
    default_dict = {
        "etype": "blstmp",
        "elayers": 4,
        "eunits": 300,
        "subsample": "1",
        "dropout_rate": 0.0,
        "atype": "dot",
        "adim": 320,
        "awin": 5,
        "aheads": 4,
        "aconv_chans": -1,
        "aconv_filts": 100,
    }
    for k in default_dict.keys():
        if isinstance(vars(args)[k], list):
            if len(vars(args)[k]) != args.num_encs:
                logging.warning(
                    "Length mismatch {}: Convert {} to {}.".format(
                        k, vars(args)[k], vars(args)[k][: args.num_encs]
                    )
                )
            vars(args)[k] = vars(args)[k][: args.num_encs]
        else:
            if not vars(args)[k]:
                # assign default value if it is None
                vars(args)[k] = default_dict[k]
                logging.warning(
                    "{} is not specified, use default value {}.".format(
                        k, default_dict[k]
                    )
                )
            # duplicate
            logging.warning(
                "Type mismatch {}: Convert {} to {}.".format(
                    k, vars(args)[k], [vars(args)[k] for _ in range(args.num_encs)]
                )
            )
            vars(args)[k] = [vars(args)[k] for _ in range(args.num_encs)]
    return args
