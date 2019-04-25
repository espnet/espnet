# Copyright 2017 Johns Hopkins University (Shinji Watanabe), Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import json
import logging
import os

import torch
import torch.distributed
import torch.multiprocessing
import torch.utils.data
import torch.utils.data.distributed

from espnet.utils.deterministic_utils import set_deterministic_pytorch


def load_batchset(json_path, args):
    from espnet.asr.asr_utils import make_batchset

    with open(json_path, 'rb') as f:
        utts = json.load(f)['utts']
    return make_batchset(
        utts, args.batch_size,
        args.maxlen_in, args.maxlen_out, args.minibatches,
        min_batch_size=args.ngpu if args.ngpu > 1 else 1,
        shortest_first=args.sortagrad == -1 or args.sortagrad > 0)


def load_rnnlm(args):
    from espnet.asr.asr_utils import get_model_conf
    import espnet.lm.pytorch_backend.lm as lm_pytorch

    if args.rnnlm is not None:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(args.char_list), rnnlm_args.layer, rnnlm_args.unit))
        torch.load(args.rnnlm, rnnlm)
        return rnnlm


def save_config(args, idim, odim):
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


def build_model(args):
    from espnet.nets.pytorch_backend.e2e_asr import E2E

    # get input and output dimension info
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]['input'][0]['shape'][1])
    odim = int(valid_json[utts[0]]['output'][0]['shape'][1])
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

    # specify model architecture
    model = E2E(idim, odim, args, use_chainer_reporter=False)
    model.rnnlm = load_rnnlm(args)
    if args.rank == 0:
        save_config(args, idim, odim)
    return model


def build_optimizer(args, model):
    # Setup an optimizer
    if args.opt == 'adadelta':
        return torch.optim.Adadelta(
            model.parameters(), rho=0.95, eps=args.eps,
            weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        return torch.optim.Adam(
            model.parameters(), weight_decay=args.weight_decay)
    assert False, 'unknown optimizer: ' + args.opt


def train(args):
    set_deterministic_pytorch(args)

    # slurm available
    import os
    if args.world_size != 0 and "SLURM_NPROCS" in os.environ:
        args.world_size = int(os.environ["SLURM_NPROCS"])
        args.rank = int(os.environ["SLURM_PROCID"])
        jobid = os.environ["SLURM_JOBID"]
        args.dist_url = "file://{}.{}".format(os.path.realpath(args.outdir + "/" + args.dist_file), jobid)
        logging.info("dist-url:{} at PROCID {} / {}".format(args.dist_url, args.rank, args.world_size))
    if args.world_size > 0:
        assert args.dist_url is not None, "you need to set --dist-url manually"

    # args.ngpu = torch.cuda.device_count()
    setattr(args, "distributed", args.world_size > 0)
    if args.distributed:
        args.world_size = args.ngpu * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        torch.multiprocessing.spawn(main_worker, nprocs=args.ngpu, args=(args,))
    else:
        # Simply call main_worker function
        main_worker(0, args)


def main_worker(gpuid, args):
    """Train with the given args

    :param Namespace args: The program arguments
    """
    from espnet.asr.pytorch_backend.asr_job import PlotAttentionJob
    from espnet.asr.pytorch_backend.asr_job import TrainingJob
    from espnet.asr.pytorch_backend.asr_job import ValidationJob
    from espnet.asr.pytorch_backend.dataset import ASRDataset
    from espnet.utils.deterministic_utils import set_deterministic_pytorch
    from espnet.utils.training.job import JobRunner

    if args.distributed:
        logging.info('use GPU {} for training'.format(gpuid))
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        args.rank = args.rank * args.ngpu + gpuid
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                             world_size=args.world_size, rank=args.rank)

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    model = build_model(args)
    subsampling_factor = model.subsample[0]

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.world_size))
        torch.cuda.set_device(gpuid)
        model.cuda(gpuid)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpuid])
    elif args.ngpu > 1:
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.ngpu))
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        model.cuda()
        args.batch_size *= args.ngpu

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    model = model.to(device)

    optimizer = build_optimizer(args, model)

    # load datasets
    train_dataset = ASRDataset(load_batchset(args.train_json, args),
                               subsampling_factor, args.preprocess_conf)
    valid_dataset = ASRDataset(load_batchset(args.valid_json, args),
                               subsampling_factor, args.preprocess_conf)

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = None
    # batch_size=1 because minibatch is already made in dataset
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=sampler, num_workers=args.n_iter_processes, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, shuffle=False, num_workers=args.n_iter_processes, pin_memory=True)


    # define jobs if rank != 0, we skip training plot and validation to avoid multi-processing data race
    jobs = []
    jobs.append(TrainingJob(model, optimizer, train_loader, device=device,
                            grad_clip=args.grad_clip, dump=args.rank == 0))

    def save_model():
        m = model.module if hasattr(model, "module") else model
        torch.save(m, args.outdir + "/model." + args.criterion + ".best")

    def adjust_optimizer():
        if args.opt == 'adadelta':
            for p in optimizer.param_groups:
                p['eps'] *= args.eps_decay
                logging.info('adadelta eps decayed to ' + str(p['eps']))

    if args.rank == 0:
        jobs.append(ValidationJob(model=model, loader=valid_loader, outdir=args.outdir,
                                  patience=args.patience, criterion=args.criterion,
                                  improve_hook=save_model, device=device,
                                  no_improve_hook=adjust_optimizer))

        if args.num_save_attention > 0 and args.mtlalpha != 1.0:
            if hasattr(model, "module"):
                att_vis_fn = model.module.calculate_all_attentions
            else:
                att_vis_fn = model.calculate_all_attentions
            jobs.append(PlotAttentionJob(args.valid_json, att_vis_fn, args.outdir + "/att_ws",
                                         args.num_save_attention, device,
                                         subsampling_factor, args.preprocess_conf))

    runner = JobRunner(jobs, args.outdir, args.epochs)
    if args.resume:
        runner.load_state_dict(torch.load(args.resume))
    runner.run()
