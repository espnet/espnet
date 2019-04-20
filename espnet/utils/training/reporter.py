# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import print_function

import contextlib
import json
import time


def plot_seq(group, path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot

    fig, ax = pyplot.subplots()
    for legend, d in group:
        for k, xs in d.items():
            ax.plot(range(len(xs)), xs, label=k + "/" + legend, marker="x")

    leg = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.grid()
    pyplot.xlabel('epoch')
    fig.savefig(path, bbox_extra_artists=(leg,), bbox_inches='tight')
    pyplot.close()


def default_print(*args, **kwargs):
    print(*args, flush=True, **kwargs)


class Stats(object):
    def __init__(self, max_epoch, outdir=None, float_fmt="{}",
                 report_every=100, logfun=default_print, optional=dict()):
        self.outdir = outdir
        if outdir is not None:
            self.log_path = outdir + "/log"
            with open(self.log_path, "w") as f:
                json.dump([], f)
        else:
            self.log_path = None
        self.logfun = logfun
        self.max_epoch = max_epoch
        self.report_every = report_every
        self.float_fmt = float_fmt
        self.start_time = time.time()
        self.current_epoch = 0
        self.plot_dict = dict()

    def state_dict(self):
        return dict(current_epoch=self.current_epoch,
                    plot_dict=self.plot_dict)

    def load_state_dict(self, states):
        self.current_epoch = states["current_epoch"]
        self.plot_dict = states["plot_dict"]

    def elapsed_time(self):
        return time.time() - self.start_time

    @contextlib.contextmanager
    def epoch(self, prefix):
        from collections import defaultdict

        try:
            e_result = EpochStats(self, prefix)
            yield e_result
        finally:
            self.logfun("[{}] epoch: {}\t{}".format(
                prefix, self.current_epoch, e_result.summary()))
            e_result.dump()

            if self.outdir is not None:
                avg = e_result.average()
                groups = defaultdict(list)
                for k, v in avg.items():
                    if k not in self.plot_dict.keys():
                        self.plot_dict[k] = dict()
                    if prefix not in self.plot_dict[k].keys():
                        self.plot_dict[k][prefix] = []
                    self.plot_dict[k][prefix].append(avg[k])
                    # gather prefix e.g., loss_ctc -> loss
                    gk = k.split('_')[0]
                    groups[gk].append((k, self.plot_dict[k]))

                for gk, gv in groups.items():
                    plot_seq(gv, self.outdir + "/" + gk + ".png")


class EpochStats(object):
    def __init__(self, global_result, prefix):
        self.global_result = global_result
        self.sum_dict = dict()
        self.iteration = 0
        self.logfun = global_result.logfun
        self.log_path = global_result.log_path
        self.prefix = prefix
        self.float_fmt = global_result.float_fmt

    def summary(self):
        s = ""
        fmt = "{}: " + self.float_fmt + "\t"
        for k, v in self.average().items():
            s += fmt.format(k, v)
        s += "elapsed: " + \
            time.strftime("%X", time.gmtime(self.global_result.elapsed_time()))
        return s

    def dump(self):
        if self.log_path is None:
            return

        with open(self.log_path, "r") as f:
            d = json.load(f)

        elem = {
            "epoch": self.global_result.current_epoch,
            "iteration": self.iteration,
            "elapsed_time": self.global_result.elapsed_time()
        }

        for k, v in self.average().items():
            elem[self.prefix + "/" + k] = v

        d.append(elem)
        with open(self.log_path, "w") as f:
            json.dump(d, f, indent=4)

    def report(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if k not in self.sum_dict.keys():
                self.sum_dict[k] = float(v)
            else:
                self.sum_dict[k] += float(v)
        self.iteration += 1
        if self.iteration % self.global_result.report_every == 0:
            self.logfun(
                "iter: {}\t{}".format(self.iteration, self.summary()))
            self.dump()

    def average(self):
        return {k: v / self.iteration for k, v in self.sum_dict.items()}
