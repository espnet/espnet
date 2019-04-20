# Copyright 2017 Johns Hopkins University (Shinji Watanabe), Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


REPORT_INTERVAL = 100


def load_batchset(json_path, args):
    import json
    from espnet.asr.asr_utils import make_batchset

    with open(json_path, 'rb') as f:
        utts = json.load(f)['utts']
    return make_batchset(
        utts, args.batch_size,
        args.maxlen_in, args.maxlen_out, args.minibatches,
        min_batch_size=args.ngpu if args.ngpu > 1 else 1,
        shortest_first=args.sortagrad == -1 or args.sortagrad > 0)


def train(args):
    """Train with the given args

    :param Namespace args: The program arguments
    """
    import json
    import os
    import torch
    import logging

    from espnet.asr.asr_utils import get_model_conf
    from espnet.asr.pytorch_backend.asr_job import ASRDataset
    from espnet.asr.pytorch_backend.asr_job import PlotAttentionJob
    from espnet.asr.pytorch_backend.asr_job import TrainingJob
    from espnet.asr.pytorch_backend.asr_job import ValidationJob
    from espnet.nets.pytorch_backend.e2e_asr import E2E
    from espnet.utils.training.job import JobRunner
    from espnet.utils.deterministic_utils import set_deterministic_pytorch

    import espnet.lm.pytorch_backend.lm as lm_pytorch

    set_deterministic_pytorch(args)

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get input and output dimension info
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]['input'][0]['shape'][1])
    odim = int(valid_json[utts[0]]['output'][0]['shape'][1])
    logging.info('#input dims : ' + str(idim))
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
    model = E2E(idim, odim, args, use_chainer_reporter=False)
    subsampling_factor = model.subsample[0]

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
        f.write(json.dumps((idim, odim, vars(args)),
                           indent=4, sort_keys=True).encode('utf_8'))
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # TODO(karita) support distributed data parallel
    # check the use of multi-gpu
    if args.ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.ngpu))
        args.batch_size *= args.ngpu

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    model = model.to(device)

    # Setup an optimizer
    if args.opt == 'adadelta':
        optimizer = torch.optim.Adadelta(
            model.parameters(), rho=0.95, eps=args.eps,
            weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     weight_decay=args.weight_decay)

    # load datasets
    train_dataset = ASRDataset(
        load_batchset(args.train_json, args),
        subsampling_factor, args.preprocess_conf)
    valid_dataset = ASRDataset(
        load_batchset(args.valid_json, args),
        subsampling_factor, args.preprocess_conf)

    # TODO(karita) support distributed sampler & loader
    # see also: https://github.com/pytorch/examples/blob/master/imagenet/main.py

    # if distributed:
    #      sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1,  # NOTE: minibatch is already made in dataset
        shuffle=sampler, num_workers=args.n_iter_processes, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1,  # NOTE: minibatch is already made in dataset
        shuffle=False, num_workers=args.n_iter_processes, pin_memory=True)

    train_job = TrainingJob(model, optimizer, train_loader, grad_clip=args.grad_clip)

    def save_model():
        m = model.module if hasattr(model, "module") else model
        torch.save(m, args.outdir + "/model." + args.criterion + ".best")

    def adjust_optimizer():
        if args.opt == 'adadelta':
            for p in optimizer.param_groups:
                p['eps'] *= args.eps_decay
                logging.info('adadelta eps decayed to ' + str(p['eps']))

    # TODO(karita) improve_hook and no_improve_hook
    valid_job = ValidationJob(model=model, loader=valid_loader, outdir=args.outdir,
                              patience=args.patience, criterion=args.criterion,
                              improve_hook=save_model,
                              no_improve_hook=adjust_optimizer)

    # Save attention weight each epoch
    if args.num_save_attention > 0 and args.mtlalpha != 1.0:
        if hasattr(model, "module"):
            att_vis_fn = model.module.calculate_all_attentions
        else:
            att_vis_fn = model.calculate_all_attentions
        plot_job = PlotAttentionJob(valid_json, att_vis_fn, args.outdir + "/att_ws",
                                    args.num_save_attention, subsampling_factor,
                                    args.preprocess_conf)
    else:
        plot_job = None

    runner = JobRunner([train_job, valid_job, plot_job], args.outdir, args.epochs)
    if args.resume:
        runner.load_state_dict(torch.load(args.resume))
    runner.run()
