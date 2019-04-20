import logging

import torch
import torch.utils.data

from espnet.utils.training.job import Job


def squeeze(data):
    return [d.squeeze(0) for d in data]


class TrainingJob(Job):
    '''Training Job

    :param espnet.nets.pytorch_backend.e2e_asr.E2E model: model to be trained
    :param torch.optim.Optimizer optimizer: optimizer for training
    :param iterable loader: dataset loader
    :param int max_epoch: max epoch for training
    '''

    def __init__(self, model, optimizer, loader,
                 grad_clip=float('inf'), accum_grad=1):
        self.model = model
        self.optimizer = optimizer
        self.loader = loader
        self.accum_grad = accum_grad
        self.grad_clip = grad_clip

    def run(self, stats):
        import math

        self.model.train()
        n_accum = 0
        with stats.epoch("main") as train_stat:
            self.model.reporter = train_stat
            for i, data in enumerate(self.loader):
                loss = self.model(*squeeze(data))[0]
                loss.mean().backward()
                n_accum += 1
                if n_accum == self.accum_grad:
                    n_accum = 0
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip
                    )
                    logging.info('grad norm={}'.format(grad_norm))
                    if math.isnan(grad_norm):
                        logging.warning('grad norm is nan. Do not update model.')
                    else:
                        self.optimizer.step()
                        self.model.zero_grad()
                        train_stat.report

    def state_dict(self):
        return dict(model=self.model.state_dict(),
                    optimizer=self.optimizer.state_dict())


def less(best, curr):
    return best > curr


def greater(best, curr):
    return best < curr


class ValidationJob(Job):
    '''Validation Job

    :param espnet.nets.pytorch_backend.e2e_asr.E2E model:
    model to be evaluated
    :param iterable loader: dataset loader
    :param int patience: max count for no improvement until termination
    :param str criterion: criterion to measure improvement
    :param function compare_fn: compare best and current criteria to check improvement:
        e.g. lambda best, curr: best < curr
    :param function improve_hook: hook called on improvement
    :param function no_improve_hook: hook called on no improvement
    '''

    def __init__(self, model, loader, outdir, patience,
                 criterion='acc', init_criterion=None, compare_fn=None,
                 improve_hook=lambda: None, no_improve_hook=lambda: None):
        self.model = model
        self.loader = loader
        self.outdir = outdir
        self.patience = patience
        self.criterion = criterion
        self.compare_fn, self.best = self.setup(compare_fn, init_criterion)

        self.improve_hook = improve_hook
        self.no_improve_hook = no_improve_hook

        # stats
        self.no_improve = 0

    def setup(self, compare_fn, init_criterion):
        if compare_fn is not None or init_criterion is not None:
            assert compare_fn is not None, 'need to set this with init_criterion'
            assert init_criterion is not None, 'need to sett this compare_fn'
            return compare_fn, init_criterion

        if self.criterion.startswith('acc'):
            return greater, -float('inf')
        else:
            for prefix in ('loss', 'cer', 'wer'):
                if self.criterion.startswith(prefix):
                    return less, float('inf')
        assert False, 'manually set compare_fn and init_criterion for ' + self.criterion

    def run(self, stats):
        self.model.eval()
        with stats.epoch("validation/main") as valid_stat:
            self.model.reporter = valid_stat
            with torch.no_grad():
                for i, data in enumerate(self.loader):
                    self.model(*squeeze(data))

        # TODO(karita) summary report, user-defined criterion
        curr = valid_stat.average()[self.criterion]
        if self.compare_fn(self.best, curr):
            self.best = curr
            self.no_improve = 0
            self.improve_hook()
        else:
            self.no_improve += 1
            self.no_improve_hook()

    def terminate(self, stats):
        return self.patience < self.no_improve

    def state_dict(self):
        return dict(patience=self.patience,
                    best=self.best,
                    no_improve=self.no_improve)


class PlotAttentionJob(Job):
    '''Plot attention matrix after Validation Job

    :param ValidationJob valid_job: validation job before this job
    :param function att_viz_fn: function to visualize attention
    :param str outdir: output dir of plots
    :param int num_save_attention: the number of saving attention plots
    :param int subsampling_factor: preprocessing parameter (TODO(karita) remove this)
    :param str preprocess_conf: path to preproces conf file
    '''

    def __init__(self, valid_json, att_vis_fn, outdir, num_save_attention,
                 subsampling_factor=0, preprocess_conf=None, reverse=False):
        import os

        self.att_vis_fn = att_vis_fn
        self.outdir = outdir
        self.data = sorted(list(valid_json.items())[:num_save_attention],
                           key=lambda x: int(x[1]['input'][0]['shape'][1]),
                           reverse=True)
        self.dataset = ASRDataset([self.data], subsampling_factor, preprocess_conf)
        self.reverse = reverse
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def run(self, stats):
        # TODO(karita) support tensorboard without chainer
        att_ws = self.att_vis_fn(*self.dataset[0])
        for idx, att_w in enumerate(att_ws):
            filename = "%s/%s.ep.%d.png" % (
                self.outdir, self.data[idx][0], stats.current_epoch)
            att_w = self.get_attention_weight(idx, att_w)
            self._plot_and_save_attention(att_w, filename)

    def get_attention_weight(self, idx, att_w):
        if self.reverse:
            dec_len = int(self.data[idx][1]['input'][0]['shape'][0])
            enc_len = int(self.data[idx][1]['output'][0]['shape'][0])
        else:
            dec_len = int(self.data[idx][1]['output'][0]['shape'][0])
            enc_len = int(self.data[idx][1]['input'][0]['shape'][0])
        if len(att_w.shape) == 3:
            att_w = att_w[:, :dec_len, :enc_len]
        else:
            att_w = att_w[:dec_len, :enc_len]
        return att_w

    def draw_attention_plot(self, att_w):
        import matplotlib.pyplot as plt
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

    def _plot_and_save_attention(self, att_w, filename):
        plt = self.draw_attention_plot(att_w)
        plt.savefig(filename)
        plt.close()


class ASRDataset(torch.utils.data.Dataset):
    def __init__(self, batchset, subsampling_factor=0, preprocess_conf=None):
        from espnet.utils.io_utils import LoadInputsAndTargets

        self.batchset = batchset
        self.subsampling_factor = subsampling_factor
        self.load_inputs_and_targets = LoadInputsAndTargets(
            mode='asr', load_output=True, preprocess_conf=preprocess_conf)
        self.ignore_id = -1

    def __getitem__(self, index):
        from torch.nn.utils.rnn import pad_sequence

        xs, ys = self.load_inputs_and_targets(self.batchset[index])

        # TODO(karita) make this subsampling inside model
        # perform subsampling
        if self.subsampling_factor > 1:
            xs = [x[::self.subsampling_factor, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = torch.tensor([x.shape[0] for x in xs])

        # perform padding and convert to tensor
        xs_pad = pad_sequence([torch.tensor(x).float() for x in xs],
                              batch_first=True, padding_value=0)
        ys_pad = pad_sequence([torch.tensor(y).long() for y in ys],
                              batch_first=True, padding_value=self.ignore_id)
        return xs_pad, ilens, ys_pad

    def __len__(self):
        return len(self.batchset)
