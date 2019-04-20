from espnet.bin.asr_train import get_parser
from espnet.bin.asr_train import init_env


def debug_args():
    import os.path
    from tempfile import mkdtemp

    test_root = os.path.dirname(os.path.realpath(__file__))
    tmpdir = "tmp" # mkdtemp('espnet')
    print('tmpdir:', tmpdir)
    # retrieved from egs/an4/asr1/run.sh
    s = '--ngpu 0 --backend pytorch --outdir ' + tmpdir + '/results --tensorboard-dir ' + tmpdir + '/tensorboard --debugmode 1 --dict ' + test_root + '/an4sub_dict.txt --debugdir ' + tmpdir + ' --minibatches 0 --verbose 0 --resume --train-json ' + test_root + '/an4sub.json --valid-json ' + test_root + '/an4sub.json --etype blstmp --elayers 4 --eunits 320 --eprojs 320 --subsample 1_2_2_1_1 --dlayers 1 --dunits 300 --atype location --adim 320 --aconv-chans 10 --aconv-filts 100 --mtlalpha 0.5 --batch-size 30 --maxlen-in 800 --maxlen-out 150 --sampling-probability 0.0 --opt adadelta --sortagrad 0 --epochs 20 --patience 3'  # noqa
    return get_parser().parse_args(s.split())


def test_trainer():
    import espnet.asr.pytorch_backend.train as A

    args = debug_args()
    init_env(args)
    A.train(args)


test_trainer()
