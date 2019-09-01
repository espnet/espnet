import logging
import os
import torch

from collections import OrderedDict

from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.mt_interface import MTInterface

from espnet.utils.dynamic_import import dynamic_import


def transfer_verification(model_state_dict, partial_state_dict, modules):
    """Verify tuples (key, shape) for input model modules match specified modules.

    Args:
        model_state_dict (odict): the initial model state_dict
        partial_state_dict (odict): the trained model state_dict
        modules (list): specified module list for transfer

    Return:
        (boolean): allow transfer
    """

    modules_model = []
    partial_modules = []

    for key_p, value_p in partial_state_dict.items():
        if any(key_p.startswith(m) for m in modules):
            partial_modules += [(key_p, value_p.shape)]

    for key_m, value_m in model_state_dict.items():
        if any(key_m.startswith(m) for m in modules):
            modules_model += [(key_m, value_m.shape)]

    len_match = (len(modules_model) == len(partial_modules))

    module_match = (sorted(modules_model, key=lambda x: (x[0], x[1])) ==
                    sorted(partial_modules, key=lambda x: (x[0], x[1])))

    return len_match and module_match


def get_partial_asr_mt_state_dict(model_state_dict, modules):
    """Create state_dict with specified modules matching input model modules.

    Args:
        model_state_dict (odict): trained model state_dict
        modules (list): specified module list for transfer

    Return:
        new_state_dict (odict): the updated state_dict
    """

    new_state_dict = OrderedDict()

    for key, value in model_state_dict.items():
        if any(key.startswith(m) for m in modules):
            new_state_dict[key] = value

    return new_state_dict


def get_partial_lm_state_dict(model_state_dict, modules):
    """Create compatible ASR state_dict from model_state_dict (LM).

    The keys for specified modules are modified to match ASR decoder modules keys.

    Args:
        model_state_dict (odict): trained model state_dict
        modules (list): specified module list for transfer

    Return:
        new_state_dict (odict): the updated state_dict
        new_mods (list): the updated module list
    """

    new_state_dict = OrderedDict()
    new_modules = []

    for key, value in list(model_state_dict.items()):
        if key == "predictor.embed.weight" \
           and "predictor.embed." in modules:
            new_key = "dec.embed.weight"
            new_state_dict[new_key] = value
            new_modules += [new_key]
        elif "predictor.rnn." in key \
             and "predictor.rnn." in modules:
            new_key = "dec.decoder." + key.split("predictor.rnn.", 1)[1]
            new_state_dict[new_key] = value
            new_modules += [new_key]

    return new_state_dict, new_modules


def filter_modules(model_state_dict, modules):
    """Filter non-matched modules in module_state_dict

    Args:
        model_state_dict (odict): trained model state_dict
        modules (list): specified module list for transfer

    Return:
        new_mods (list): the update module list
    """

    new_mods = []
    incorrect_mods = []

    mods_model = list(model_state_dict.keys())
    for mod in modules:
        if any(key.startswith(mod) for key in mods_model):
            new_mods += [mod]
        else:
            incorrect_mods += [mod]

    if incorrect_mods:
        logging.info("module(s) %s don\'t match or (partially match) "
                     "available modules in model.", incorrect_mods)
        logging.info('for information, the existing modules in model are:')
        logging.info('%s', mods_model)

    return new_mods


def load_trained_model(model_path):
    """Load the trained model for recognition.

    Args:
        model_path(str): Path to model.***.best
    """

    idim, odim, train_args = get_model_conf(
        model_path, os.path.join(os.path.dirname(model_path), 'model.json'))

    logging.info('reading model parameters from ' + model_path)

    if hasattr(train_args, "model_module"):
        model_module = train_args.model_module
    else:
        model_module = "espnet.nets.pytorch_backend.e2e_asr:E2E"
    model_class = dynamic_import(model_module)
    model = model_class(idim, odim, train_args)
    torch_load(model_path, model)

    return model, train_args


def get_trained_model_state_dict(model_path):
    """Extract the trained model state dict for pre-initialization.

    Args:
        model_path (str): Path to model.***.best

    Return:
        model.state_dict() (odict): the loaded model state_dict
        (str): Type of model. Either ASR/MT or LM.
    """

    conf_path = os.path.join(os.path.dirname(model_path), 'model.json')
    if 'rnnlm' in model_path:
        logging.info('reading model parameters from %s', model_path)

        return torch.load(model_path), 'lm'

    idim, odim, args = get_model_conf(model_path, conf_path)

    logging.info('reading model parameters from ' + model_path)

    if hasattr(args, "model_module"):
        model_module = args.model_module
    else:
        model_module = "espnet.nets.pytorch_backend.e2e_asr:E2E"

    model_class = dynamic_import(model_module)
    model = model_class(idim, odim, args)
    torch_load(model_path, model)
    assert isinstance(model, MTInterface) or isinstance(model, ASRInterface)

    return model.state_dict(), 'asr-mt'


def load_trained_modules(idim, odim, args):
    """Load model encoder or/and decoder modules with ESPNET pre-trained model(s).

    Args:
        idim (int): initial input dimension.
        odim (int): initial output dimension.
        args (namespace): The initial model arguments.

    Return:
        model (torch.nn.Module): The model with pretrained modules.
    """

    enc_model_path = args.enc_init
    dec_model_path = args.dec_init
    enc_modules = args.enc_init_mods
    dec_modules = args.dec_init_mods

    model_class = dynamic_import(args.model_module)
    main_model = model_class(idim, odim, args)
    assert isinstance(main_model, ASRInterface)

    main_state_dict = main_model.state_dict()

    logging.info('model(s) found for pre-initialization')
    for model_path, modules in [(enc_model_path, enc_modules),
                                (dec_model_path, dec_modules)]:
        if model_path is not None:
            if os.path.isfile(model_path):
                model_state_dict, mode = get_trained_model_state_dict(model_path)

                modules = filter_modules(model_state_dict, modules)
                if mode == 'lm':
                    partial_state_dict, modules = get_partial_lm_state_dict(model_state_dict, modules)
                else:
                    partial_state_dict = get_partial_asr_mt_state_dict(model_state_dict, modules)

                    if partial_state_dict:
                        if transfer_verification(main_state_dict, partial_state_dict,
                                                 modules):
                            logging.info('loading %s from model: %s', modules, model_path)
                            main_state_dict.update(partial_state_dict)
                        else:
                            logging.info('modules %s in model %s don\'t match your training config',
                                         modules, model_path)
            else:
                logging.info('model was not found : %s', model_path)

    main_model.load_state_dict(main_state_dict)

    return main_model


def load_trained_modules_mulenc(model, process):
    """Load the pre-trained modules.

    Args:
        model(torch.nn.Module): torch model
        process (dict): dictionary to define pretrained modules to load.
                        process.keys=['prefix_src', 'prefix_tgt', 'trained_model', 'freeze_params']
                        prefix_src: prefix of modules to be used in the pre-trained model
                        prefix_tgt: prefix of modules to be updated with in the target model
                        trained_model: path of the pre-trained model
                        freeze_params: flag to freeze the modules during training
                        process=
                        {'prefix_src': 'enc',
                        'prefix_tgt': 'enc.0',
                        'trained_model': exp/exp_name/results/model.acc.best,
                        'freeze_params': true}
    """
    for key in ['prefix_src', 'prefix_tgt', 'trained_model']:
        assert key in process, '{} is not specified.'.format(key)

    prefix_src = process['prefix_src']
    prefix_tgt = process['prefix_tgt']
    trained_model = process['trained_model']
    freeze_params = process['freeze_params'] if 'freeze_params' in process and process['freeze_params'] is True else False

    if 'snapshot' in trained_model:
        model_state_dict = torch.load(trained_model, map_location=lambda storage, loc: storage)['model']
    else:
        model_state_dict = torch.load(trained_model, map_location=lambda storage, loc: storage)

    # change prefix_src to prefix_tgt in model_state_dict
    model_state_dict = dict(model_state_dict)
    keys = [*model_state_dict.keys()]
    for key in keys:
        if key.startswith(prefix_src):
            key_tgt = key.replace(prefix_src, prefix_tgt, 1)
        else:
            key_tgt = key.replace(key[:2], 'dummy.', 1)
        model_state_dict[key_tgt] = model_state_dict.pop(key)

    # filter
    param_dict = dict(model.named_parameters())
    model_state_dict_filtered = {k: v for k, v in model_state_dict.items() if k in param_dict.keys() and v.size() == model.state_dict()[k].size()}

    logging.warning("Loading pretrained modules. prefix_src:{}; prefix_tgt:{}".format(prefix_src, prefix_tgt))
    for key in model_state_dict_filtered.keys():
        logging.warning("Inititializing {}".format(key))

    if hasattr(model, 'module'):
        model.module.load_state_dict(model_state_dict_filtered, strict=False)
    else:
        model.load_state_dict(model_state_dict_filtered, strict=False)

    if freeze_params:
        for name, param in model.named_parameters():
            if name in model_state_dict_filtered.keys():
                param.requires_grad = False
                logging.warning("Freezed {}".format(name))

    del model_state_dict, model_state_dict_filtered, param_dict