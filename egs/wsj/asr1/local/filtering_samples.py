#!/usr/bin/env python3

# Copyright 2019 Shanghai Jiao Tong University (Wangyou Zhang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import json
import math
import sys

from chainer import training
from chainer.training.updater import StandardUpdater
import torch

from espnet.asr.asr_utils import format_mulenc_args
from espnet.nets.asr_interface import ASRInterface
from espnet.utils.dataset import ChainerDataLoader
from espnet.utils.dataset import TransformDataset
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_batchset

from espnet.asr.pytorch_backend.asr import CustomConverter
from espnet.asr.pytorch_backend.asr import CustomConverterMulEnc
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.bin.asr_train import get_parser


class CustomUpdater(StandardUpdater):
    """Custom Updater for Pytorch.

    Args:
        model (torch.nn.Module): The model to update.
        train_iter (chainer.dataset.Iterator): The training iterator.
        optimizer (torch.optim.optimizer): The training optimizer.

        device (torch.device): The device to use.
        ngpu (int): The number of gpus to use.
        use_apex (bool): The flag to use Apex in backprop.

    """

    def __init__(self, model, train_iter,
                 optimizer, device, ngpu, model_module):
        super(CustomUpdater, self).__init__(train_iter, optimizer)
        self.model = model
        self.device = device
        self.ngpu = ngpu
        self.iteration = 0
        self.invalid_samples = []
        if 'transformer' in model_module:
            self.model_type = 'transformer'
        elif 'mulenc' in model_module:
            self.model_type = 'mulenc'
        else:
            self.model_type = 'rnn'

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        """Main update routine of the CustomUpdater."""
        train_iter = self.get_iterator('main')

        # Get the next batch ( a list of json files)
        xs_pad, ilens, ys_pad = train_iter.next()

        # Compute the loss at this time step and accumulate it
        self.model.eval()

        # no gradient and backward
        with torch.no_grad():
            if self.model_type == 'rnn':
                # 0. Frontend
                if getattr(self.model, 'frontend', None) is not None:
                    hs_pad, hlens, mask = self.model.frontend(to_torch_tensor(xs_pad), ilens)
                    hs_pad, hlens = self.model.feature_transform(hs_pad, hlens)
                else:
                    hs_pad, hlens = xs_pad, ilens

                # 1. Encoder
                hs_pad, hlens, _ = self.model.enc(hs_pad, hlens)

                # 2. CTC loss
                loss = self.model.ctc(hs_pad, hlens, ys_pad)
            elif self.model_type == 'mulenc':
                loss_ctc_list = []
                for idx in range(self.model.num_encs):
                    # 1. Encoder
                    hs_pad, hlens, _ = self.model.enc[idx](xs_pad[idx], ilens[idx])

                    # 2. CTC loss
                    ctc_idx = 0 if self.model.share_ctc else idx
                    loss_ctc = self.model.ctc[ctc_idx](hs_pad, hlens, ys_pad)
                    loss_ctc_list.append(loss_ctc)

                loss = torch.sum(torch.cat(
                    [(item * self.model.weights_ctc_train[i]).unsqueeze(0) for i, item in enumerate(loss_ctc_list)]))
            elif self.model_type == 'transformer':
                # 1. Encoder
                xs_pad = xs_pad[:, :max(ilens)]  # for data parallel
                src_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)
                hs_pad, hs_mask = self.model.encoder(xs_pad, src_mask)

                # 2. CTC loss
                batch_size = xs_pad.size(0)
                hs_len = hs_mask.view(batch_size, -1).sum(1)
                loss = self.model.ctc(hs_pad.view(batch_size, -1, self.model.adim), hs_len, ys_pad)
            else:
                raise NameError('Unsupported model type: {}'.format(self.model_type))

        loss.detach_()  # Truncate the graph
        loss_data = float(loss)

        if loss_data >= CTC_LOSS_THRESHOLD or math.isnan(loss_data):
            self.invalid_samples.append(self.iteration)

    def update(self):
        self.update_core()
        self.iteration += 1


def filtering_train_json(train_json, sample_ids):
    """Filtering out the invalid samples from the original train_json.

    Args:
        train_json (dict): Dictionary of training data.
        sample_ids (list): List of ids of samples to be filtered out.

    Returns:
        new_train_json (dict): Filtered dictionary of training data.
    """
    new_train_json = train_json.copy()
    for sample in sample_ids:
        new_train_json.pop(sample)
        print("Invalid sample '{}' is removed".format(sample))
    else:
        print("No invalid samples are detected.")
    return new_train_json


def pseudo_train(args):
    """Pretend to train with the given args to check if the training samples are valid.

    Too short samples will be detected and reported if they lead to ``loss_ctc=inf``.

   The training data will be sorted in the order of input length (long to short) and not shuffled.

    Args:
        args (namespace): The program arguments.
    """
    # overridden some arguments to save time and memory cost
    args.batch_size = 1
    args.epochs = 1
    args.lsm_type = ''
    args.rnnlm = None
    args.ngpu = 1
    args.sortagrad = 0

    set_deterministic_pytorch(args)
    if args.num_encs > 1:
        args = format_mulenc_args(args)

    # get input and output dimension info
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    utt0 = next(iter(train_json))
    idim_list = [int(train_json[utt0]['input'][i]['shape'][-1]) for i in range(args.num_encs)]
    odim = int(train_json[utt0]['output'][0]['shape'][-1])

    model_class = dynamic_import(args.model_module)
    model = model_class(idim_list[0] if args.num_encs == 1 else idim_list, odim, args)
    assert isinstance(model, ASRInterface)
    if getattr(model, 'ctc', None) is None:
        print("'E2E' object has no attribute 'ctc'. Skip this script and go ahead.")
        return None

    reporter = model.reporter

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

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # Setup a converter
    if args.num_encs == 1:
        converter = CustomConverter(subsampling_factor=model.subsample[0], dtype=dtype)
    else:
        converter = CustomConverterMulEnc([i[0] for i in model.subsample_list], dtype=dtype)

    # make minibatch list (batch_size = 1)
    # from long to short
    train = make_batchset(train_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          shortest_first=False,
                          count=args.batch_count,
                          batch_bins=args.batch_bins,
                          batch_frames_in=args.batch_frames_in,
                          batch_frames_out=args.batch_frames_out,
                          batch_frames_inout=args.batch_frames_inout,
                          iaxis=0, oaxis=0)

    load_tr = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': True}  # Switch the mode of preprocessing
    )
    # hack to make batchsize argument as 1
    # actual bathsize is included in a list
    # default collate function converts numpy array to pytorch tensor
    # we used an empty collate function instead which returns list
    train_iter = {'main': ChainerDataLoader(
        dataset=TransformDataset(train, lambda data: converter([load_tr(data)], device=device)),
        batch_size=1, num_workers=args.n_iter_processes,
        shuffle=False, collate_fn=lambda x: x[0])}

    # Set up a trainer
    updater = CustomUpdater(model, train_iter, optimizer, device, args.ngpu, args.model_module)
    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    # Run the training
    trainer.run()

    # ids of the invalid samples
    sample_ids = [train[i][0][0] for i in updater.invalid_samples]
    return filtering_train_json(train_json, sample_ids)


if __name__ == '__main__':
    cmd_args = sys.argv[1:]
    parser = get_parser(required=False)
    parser.add_argument('--output-json-path', type=str, required=True,
                        help='Output path of the filtered json file')
    args, _ = parser.parse_known_args(cmd_args)

    if args.model_module is None:
        model_module = "espnet.nets." + args.backend + "_backend.e2e_asr:E2E"
    else:
        model_module = args.model_module
    model_class = dynamic_import(model_module)
    model_class.add_arguments(parser)

    args = parser.parse_args(cmd_args)
    args.model_module = model_module
    if 'chainer_backend' in args.model_module:
        args.backend = 'chainer'
    if 'pytorch_backend' in args.model_module:
        args.backend = 'pytorch'

    # load dictionary
    if args.dict is not None:
        with open(args.dict, 'rb') as f:
            dictionary = f.readlines()
        char_list = [entry.decode('utf-8').split(' ')[0]
                     for entry in dictionary]
        char_list.insert(0, '<blank>')
        char_list.append('<eos>')
        args.char_list = char_list
    else:
        args.char_list = None

    # get filtered train_json without invalid samples
    new_train_json = pseudo_train(args)
    if new_train_json is not None:
        jsonstring = json.dumps({'utts': new_train_json}, indent=4, ensure_ascii=False, sort_keys=True)
        with open(args.output_json_path, 'w') as f:
            f.write(jsonstring)
