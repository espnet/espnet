# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import torch

from espnet.bin.asr_train import get_parser
from espnet.bin.asr_train import init_env


def debug_args(tmpdir, ngpu):
    import os.path
    # from tempfile import mkdtemp
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)

    test_root = os.path.dirname(os.path.realpath(__file__))
    # retrieved from egs/an4/asr1/run.sh
    s = '--ngpu ' + str(ngpu) + ' --backend pytorch --outdir ' + tmpdir + '/results --tensorboard-dir ' + tmpdir + '/tensorboard --debugmode 1 --dict ' + test_root + '/an4sub_dict.txt --debugdir ' + tmpdir + ' --minibatches 0 --verbose 0 --resume --train-json ' + test_root + '/an4sub.json --valid-json ' + test_root + '/an4sub.json --etype blstmp --elayers 4 --eunits 320 --eprojs 320 --subsample 1_2_2_1_1 --dlayers 1 --dunits 300 --atype location --adim 320 --aconv-chans 10 --aconv-filts 100 --mtlalpha 0.5 --batch-size 4 --maxlen-in 800 --maxlen-out 150 --sampling-probability 0.0 --opt adadelta --sortagrad 0 --epochs 6 --patience 3'  # noqa
    return get_parser().parse_args(s.split())


def run_reporter():
    import espnet.asr.pytorch_backend.asr as chainer_reporter
    import espnet.asr.pytorch_backend.train as new_reporter

    if torch.cuda.is_available():
        args_new = debug_args("tmp-new-report", 1)
        args_chainer = debug_args("tmp-chainer-report", 1)
        init_env(args_new)
        new_reporter.train(args_new)
        init_env(args_chainer)
        chainer_reporter.train(args_chainer)

    args_new = debug_args("tmp-new-report", 0)
    args_chainer = debug_args("tmp-chainer-report", 0)
    init_env(args_new)
    new_reporter.train(args_new)
    init_env(args_chainer)
    chainer_reporter.train(args_chainer)


if __name__ == '__main__':
    run_reporter()
