#!/usr/bin/env python3

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Copyright 2019 Johns Hopkins University (Ruizhi Li)

#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import json
import logging
import os
import sys
import io
import yaml

from chainer.datasets import TransformDataset
from chainer import training
from chainer.training import extensions
import numpy as np
from tensorboardX import SummaryWriter
import torch

from espnet.asr.asr_utils import adadelta_eps_decay
from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import CompareValueTrigger
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import mtlalpha_exp_decay
from espnet.asr.asr_utils import restore_snapshot
from espnet.asr.asr_utils import sampling_probability_exp_decay
from espnet.asr.asr_utils import snapshot_object
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_snapshot
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.asr.pytorch_backend.asr_init import load_trained_modules_mulenc
from espnet.asr.pytorch_backend.asr import CustomEvaluator
from espnet.asr.pytorch_backend.asr import CustomUpdater
import espnet.lm.pytorch_backend.extlm as extlm_pytorch
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.e2e_asr import pad_list
import espnet.nets.pytorch_backend.lm.default as lm_pytorch
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.iterators import ToggleableShufflingMultiprocessIterator
from espnet.utils.training.iterators import ToggleableShufflingSerialIterator
from espnet.utils.training.tensorboard_logger import TensorboardLogger
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop

import matplotlib

matplotlib.use('Agg')

if sys.version_info[0] == 2:
    from itertools import izip_longest as zip_longest
else:
    from itertools import zip_longest as zip_longest


class CustomConverterMulEnc(object):
    """Custom batch converter for Pytorch.

    Args:
        subsampling_factors (list): List of subsampling factors for each encoder.
        dtype (torch.dtype): Data type to convert.

    """

    def __init__(self, subsamping_factors=[1, 1], dtype=torch.float32):
        self.subsamping_factors = subsamping_factors
        self.ignore_id = -1
        self.dtype = dtype
        self.num_encs = len(subsamping_factors)

    def __call__(self, batch, device):
        """Transforms a batch and send it to a device.

        Args:
            batch (list): The batch to transform.
            device (torch.device): The device to send to.

        Returns:
            tuple( list(torch.Tensor), list(torch.Tensor), torch.Tensor)

        """
        # batch should be located in list
        assert len(batch) == 1
        xs_list = batch[0][:self.num_encs]
        ys = batch[0][-1]

        # perform subsampling
        if np.sum(self.subsamping_factors) > self.num_encs:
            xs_list = [[x[::self.subsampling_factors[i], :] for x in xs_list[i]] for i in range(self.num_encs)]

        # get batch of lengths of input sequences
        ilens_list = [np.array([x.shape[0] for x in xs_list[i]]) for i in range(self.num_encs)]

        # perform padding and convert to tensor
        # currently only support real number
        xs_list_pad = [pad_list([torch.from_numpy(x).float() for x in xs_list[i]], 0).to(device, dtype=self.dtype) for i
                       in range(self.num_encs)]

        ilens_list = [torch.from_numpy(ilens_list[i]).to(device) for i in range(self.num_encs)]
        # NOTE: this is for multi-task learning (e.g., speech translation)
        ys_pad = pad_list([torch.from_numpy(np.array(y[0]) if isinstance(y, tuple) else y).long()
                           for y in ys], self.ignore_id).to(device)

        return xs_list_pad, ilens_list, ys_pad


def train(args):
    """Train with the given args.

    Args:
        args (namespace): The program arguments.

    """
    set_deterministic_pytorch(args)

    # convert None to [default_value] * num_encs
    def format_mulenc_args(args):
        # format args for multi-encoder setup
        default_dict = {'etype': 'blstmp',
                        'elayers': 4,
                        'eunits': 300,
                        'subsample': '1',
                        'dropout_rate': 0.0,
                        'atype': 'dot',
                        'adim': 320,
                        'awin': 5,
                        'aheads': 4,
                        'aconv_chans': -1,
                        'aconv_filts': 100
                        }
        for k in default_dict.keys():
            if isinstance(vars(args)[k], list):
                assert len(vars(args)[k]) == args.num_encs, (len(vars(args)[k]),)
                vars(args)[k] = vars(args)[k][:args.num_encs]
            else:
                if not vars(args)[k]:
                    # assign default value if it is None
                    vars(args)[k] = default_dict[k]
                # duplicate
                vars(args)[k] = [vars(args)[k] for _ in range(args.num_encs)]
        return args

    args = format_mulenc_args(args)

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get input and output dimension info
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    idims = [int(valid_json[utts[0]]['input'][i]['shape'][-1]) for i in range(args.num_encs)]
    odim = int(valid_json[utts[0]]['output'][0]['shape'][-1])
    for i in range(args.num_encs):
        logging.info('stream{}: input dims : {}'.format(i + 1, idims[i]))
    logging.info('#output dims: ' + str(odim))

    # specify attention, CTC, hybrid mode
    if args.mtlalpha == 1.0:
        mtl_mode = 'ctc'
        logging.info('Pure CTC mode')
    elif args.mtlalpha == 0.0:
        mtl_mode = 'att'
        logging.info('Pure attention mode')
    else:
        mtl_mode = 'mtl'
        logging.info('Multitask learning mode')

    # specify model architecture
    model_class = dynamic_import(args.model_module)
    model = model_class(idims, odim, args)
    assert isinstance(model, ASRInterface)

    # load pretrained model
    if args.pretrain_conf is not None:
        with io.open(args.pretrain_conf, encoding='utf-8') as f:
            pretrain_conf = yaml.safe_load(f)
            assert isinstance(pretrain_conf, dict), type(pretrain_conf)
            for idx, process in enumerate(pretrain_conf):
                assert isinstance(pretrain_conf[process], dict), type(pretrain_conf[process])
                load_trained_modules_mulenc(model, pretrain_conf[process])

    if args.rnnlm is not None:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(args.char_list), rnnlm_args.layer, rnnlm_args.unit))
        torch.load(args.rnnlm, rnnlm)
        model.rnnlm = rnnlm

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(json.dumps((idims, odim, vars(args)),
                           indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    reporter = model.reporter

    # check the use of multi-gpu
    if args.ngpu > 1:
        if args.batch_size != 0:
            logging.info('batch size is automatically increased (%d -> %d)' % (
                args.batch_size, args.batch_size * args.ngpu))
            args.batch_size *= args.ngpu

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    if args.train_dtype in ("float16", "float32", "float64"):
        dtype = getattr(torch, args.train_dtype)
    else:
        dtype = torch.float32
    model = model.to(device=device, dtype=dtype)

    # Setup an optimizer
    if args.opt == 'adadelta':
        optimizer = torch.optim.Adadelta(
            model.parameters(), rho=0.95, eps=args.eps,
            weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     weight_decay=args.weight_decay)
    elif args.opt == 'noam':
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt
        optimizer = get_std_opt(model, args.adim, args.transformer_warmup_steps, args.transformer_lr)
    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)

    # setup apex.amp
    if args.train_dtype in ("O0", "O1", "O2", "O3"):
        try:
            from apex import amp
        except ImportError as e:
            logging.error(f"You need to install apex for --train-dtype {args.train_dtype}. "
                          "See https://github.com/NVIDIA/apex#linux")
            raise e
        if args.opt == 'noam':
            model, optimizer.optimizer = amp.initialize(model, optimizer.optimizer, opt_level=args.train_dtype)
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.train_dtype)
        use_apex = True
    else:
        use_apex = False

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # Setup a converter
    converter = CustomConverterMulEnc([i[0] for i in model.subsample_list], dtype=dtype)

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    # make minibatch list (variable length)
    train = make_batchset(train_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          shortest_first=use_sortagrad,
                          count=args.batch_count,
                          batch_bins=args.batch_bins,
                          batch_frames_in=args.batch_frames_in,
                          batch_frames_out=args.batch_frames_out,
                          batch_frames_inout=args.batch_frames_inout,
                          problem_utts_file=args.problem_utts_file)
    valid = make_batchset(valid_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          count=args.batch_count,
                          batch_bins=args.batch_bins,
                          batch_frames_in=args.batch_frames_in,
                          batch_frames_out=args.batch_frames_out,
                          batch_frames_inout=args.batch_frames_inout,
                          problem_utts_file=args.problem_utts_file)

    load_tr = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': True}  # Switch the mode of preprocessing
    )
    load_cv = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': False}  # Switch the mode of preprocessing
    )
    # hack to make batchsize argument as 1
    # actual bathsize is included in a list
    if args.n_iter_processes > 0:
        train_iter = ToggleableShufflingMultiprocessIterator(
            TransformDataset(train, load_tr),
            batch_size=1, n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20,
            shuffle=not use_sortagrad)
        valid_iter = ToggleableShufflingMultiprocessIterator(
            TransformDataset(valid, load_cv),
            batch_size=1, repeat=False, shuffle=False,
            n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20)
    else:
        train_iter = ToggleableShufflingSerialIterator(
            TransformDataset(train, load_tr),
            batch_size=1, shuffle=not use_sortagrad)
        valid_iter = ToggleableShufflingSerialIterator(
            TransformDataset(valid, load_cv),
            batch_size=1, repeat=False, shuffle=False)

    # Set up a trainer
    updater = CustomUpdater(
        model, args.grad_clip, train_iter, optimizer,
        converter, device, args.ngpu, args.grad_noise, args.accum_grad, use_apex=use_apex)
    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    if use_sortagrad:
        trainer.extend(ShufflingEnabler([train_iter]),
                       trigger=(args.sortagrad if args.sortagrad != -1 else args.epochs, 'epoch'))

    # Resume from a snapshot
    if args.resume:
        logging.info('resumed from %s' % args.resume)
        torch_resume(args.resume, trainer)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(CustomEvaluator(model, valid_iter, reporter, converter, device, args.ngpu))

    # expoential decay for mtlalpha
    if 0 < args.mtlalpha < 1 and 0 < getattr(args, "mtlalpha_exp_decay", 1.0) < 1:
        trainer.extend(mtlalpha_exp_decay(getattr(args, "mtlalpha_exp_decay", 1.0)),
                       trigger=(1, 'epoch'))

    # expoential decay for sampling probability
    if mtl_mode != 'ctc' and 0 < getattr(args, "sampling_probability_exp_decay", 1.0) < 1:
        trainer.extend(sampling_probability_exp_decay(getattr(args, "sampling_probability_exp_decay", 1.0)),
                       trigger=(1, 'epoch'))

    # Save attention weight each epoch
    if args.num_save_attention > 0 and args.mtlalpha != 1.0:
        data = sorted(list(valid_json.items())[:args.num_save_attention],
                      key=lambda x: int(x[1]['input'][0]['shape'][1]), reverse=True)
        if hasattr(model, "module"):
            att_vis_fn = model.module.calculate_all_attentions
            plot_class = model.module.attention_plot_class
        else:
            att_vis_fn = model.calculate_all_attentions
            plot_class = model.attention_plot_class
        att_reporter = plot_class(
            att_vis_fn, data, args.outdir + "/att_ws",
            converter=converter, transform=load_cv, device=device)
        trainer.extend(att_reporter, trigger=(1, 'epoch'))
    else:
        att_reporter = None

    # Make a plot for training and validation values
    report_keys_loss_ctc = ['main/loss_ctc{}'.format(i + 1) for i in range(model.num_encs)] + [
        'validation/main/loss_ctc{}'.format(i + 1) for i in range(model.num_encs)]
    report_keys_cer_ctc = ['main/cer_ctc{}'.format(i + 1) for i in range(model.num_encs)] + [
        'validation/main/cer_ctc{}'.format(i + 1) for i in range(model.num_encs)]
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss',
                                          'main/loss_ctc', 'validation/main/loss_ctc',
                                          'main/loss_att', 'validation/main/loss_att'] + report_keys_loss_ctc,
                                         'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/acc', 'validation/main/acc'],
                                         'epoch', file_name='acc.png'))
    trainer.extend(extensions.PlotReport(['main/cer_ctc', 'validation/main/cer_ctc'] + report_keys_loss_ctc,
                                         'epoch', file_name='cer.png'))

    # Save best models
    trainer.extend(snapshot_object(model, 'model.loss.best'),
                   trigger=training.triggers.MinValueTrigger('validation/main/loss'))
    if mtl_mode != 'ctc':
        trainer.extend(snapshot_object(model, 'model.acc.best'),
                       trigger=training.triggers.MaxValueTrigger('validation/main/acc'))

    # save snapshot which contains model and optimizer states
    trainer.extend(torch_snapshot(), trigger=(1, 'epoch'))

    # epsilon decay in the optimizer
    if args.opt == 'adadelta':
        if args.criterion == 'acc' and mtl_mode != 'ctc':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.acc.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
        elif args.criterion == 'loss':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.loss.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(args.report_interval_iters, 'iteration')))
    report_keys = ['epoch', 'iteration', 'main/loss', 'main/loss_ctc', 'main/loss_att',
                   'validation/main/loss', 'validation/main/loss_ctc', 'validation/main/loss_att',
                   'main/acc', 'validation/main/acc', 'main/cer_ctc', 'validation/main/cer_ctc',
                   'elapsed_time'] + report_keys_cer_ctc + report_keys_loss_ctc
    if args.opt == 'adadelta':
        trainer.extend(extensions.observe_value(
            'eps', lambda trainer: trainer.updater.get_optimizer('main').param_groups[0]["eps"]),
            trigger=(args.report_interval_iters, 'iteration'))
        report_keys.append('eps')
    if args.report_cer:
        report_keys.append('validation/main/cer')
    if args.report_wer:
        report_keys.append('validation/main/wer')
    trainer.extend(extensions.PrintReport(
        report_keys), trigger=(args.report_interval_iters, 'iteration'))

    trainer.extend(extensions.ProgressBar(update_interval=args.report_interval_iters))
    set_early_stop(trainer, args)

    if args.tensorboard_dir is not None and args.tensorboard_dir != "":
        trainer.extend(TensorboardLogger(SummaryWriter(args.tensorboard_dir), att_reporter),
                       trigger=(args.report_interval_iters, "iteration"))
    # Run the training
    trainer.run()
    check_early_stop(trainer, args.epochs)


def recog(args):
    """Decode with the given args.

    Args:
        args (namespace): The program arguments.
    """
    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model)
    assert isinstance(model, ASRInterface)
    model.recog_args = args

    # read rnnlm
    if args.rnnlm:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        if getattr(rnnlm_args, "model_module", "default") != "default":
            raise ValueError("use '--api v2' option to decode with non-default language model")
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(train_args.char_list), rnnlm_args.layer, rnnlm_args.unit))
        torch_load(args.rnnlm, rnnlm)
        rnnlm.eval()
    else:
        rnnlm = None

    if args.word_rnnlm:
        rnnlm_args = get_model_conf(args.word_rnnlm, args.word_rnnlm_conf)
        word_dict = rnnlm_args.char_list_dict
        char_dict = {x: i for i, x in enumerate(train_args.char_list)}
        word_rnnlm = lm_pytorch.ClassifierWithState(lm_pytorch.RNNLM(
            len(word_dict), rnnlm_args.layer, rnnlm_args.unit))
        torch_load(args.word_rnnlm, word_rnnlm)
        word_rnnlm.eval()

        if rnnlm is not None:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.MultiLevelLM(word_rnnlm.predictor,
                                           rnnlm.predictor, word_dict, char_dict))
        else:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.LookAheadWordLM(word_rnnlm.predictor,
                                              word_dict, char_dict))

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()
        if rnnlm:
            rnnlm.cuda()

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']
    new_js = {}

    load_inputs_and_targets = LoadInputsAndTargets(
        mode='asr', load_output=False, sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None else args.preprocess_conf,
        preprocess_args={'train': False})

    if args.batchsize == 0:
        with torch.no_grad():
            for idx, name in enumerate(js.keys(), 1):
                logging.info('(%d/%d) decoding ' + name, idx, len(js.keys()))
                batch = [(name, js[name])]
                feat = load_inputs_and_targets(batch)
                feat = [feat[idx][0] for idx in range(model.num_encs)]
                nbest_hyps = model.recognize(feat, args, train_args.char_list, rnnlm)
                new_js[name] = add_results_to_json(js[name], nbest_hyps, train_args.char_list)

    else:
        def grouper(n, iterable, fillvalue=None):
            kargs = [iter(iterable)] * n
            return zip_longest(*kargs, fillvalue=fillvalue)

        # sort data if batchsize > 1
        keys = list(js.keys())
        if args.batchsize > 1:
            feat_lens = [js[key]['input'][0]['shape'][0] for key in keys]
            sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
            keys = [keys[i] for i in sorted_index]

        with torch.no_grad():
            for names in grouper(args.batchsize, keys, None):
                names = [name for name in names if name]
                batch = [(name, js[name]) for name in names]
                feats = load_inputs_and_targets(batch)
                nbest_hyps = model.recognize_batch(feats, args, train_args.char_list, rnnlm=rnnlm)

                for i, nbest_hyp in enumerate(nbest_hyps):
                    name = names[i]
                    new_js[name] = add_results_to_json(js[name], nbest_hyp, train_args.char_list)

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
