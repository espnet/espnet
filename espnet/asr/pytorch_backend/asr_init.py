"""Finetuning methods."""

import logging
import os
import torch

from collections import OrderedDict

from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.mt_interface import MTInterface
from espnet.nets.tts_interface import TTSInterface

from espnet.utils.dynamic_import import dynamic_import


def freeze_modules(model, modules):
    """Freeze model parameters according to modules list.

    Args:
        model (torch.nn.Module): main model to update
        modules (list): specified module list for freezing

    Return:
        model (torch.nn.Module): updated model
        model_params (filter): filtered model parameters

    """
    for mod, param in model.named_parameters():
        if any(mod.startswith(m) for m in modules):
            logging.info(f"freezing {mod}, it will not be updated.")
            param.requires_grad = False

    model_params = filter(lambda x: x.requires_grad, model.parameters())

    return model, model_params


def transfer_verification(model_state_dict, partial_state_dict, modules):
    """Verify tuples (key, shape) for input model modules match specified modules.

    Args:
        model_state_dict (OrderedDict): the initial model state_dict
        partial_state_dict (OrderedDict): the trained model state_dict
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

    len_match = len(modules_model) == len(partial_modules)

    module_match = sorted(modules_model, key=lambda x: (x[0], x[1])) == sorted(
        partial_modules, key=lambda x: (x[0], x[1])
    )

    return len_match and module_match


def get_partial_state_dict(model_state_dict, modules):
    """Create state_dict with specified modules matching input model modules.

    Note that get_partial_lm_state_dict is used if a LM specified.

    Args:
        model_state_dict (OrderedDict): trained model state_dict
        modules (list): specified module list for transfer

    Return:
        new_state_dict (OrderedDict): the updated state_dict

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
        model_state_dict (OrderedDict): trained model state_dict
        modules (list): specified module list for transfer

    Return:
        new_state_dict (OrderedDict): the updated state_dict
        new_mods (list): the updated module list

    """
    new_state_dict = OrderedDict()
    new_modules = []

    for key, value in list(model_state_dict.items()):
        if key == "predictor.embed.weight" and "predictor.embed." in modules:
            new_key = "dec.embed.weight"
            new_state_dict[new_key] = value
            new_modules += [new_key]
        elif "predictor.rnn." in key and "predictor.rnn." in modules:
            new_key = "dec.decoder." + key.split("predictor.rnn.", 1)[1]
            new_state_dict[new_key] = value
            new_modules += [new_key]

    return new_state_dict, new_modules


def filter_modules(model_state_dict, modules):
    """Filter non-matched modules in module_state_dict.

    Args:
        model_state_dict (OrderedDict): trained model state_dict
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
        logging.warning(
            "module(s) %s don't match or (partially match) "
            "available modules in model.",
            incorrect_mods,
        )
        logging.warning("for information, the existing modules in model are:")
        logging.warning("%s", mods_model)

    return new_mods


def load_trained_model(model_path):
    """Load the trained model for recognition.

    Args:
        model_path (str): Path to model.***.best

    """
    idim, odim, train_args = get_model_conf(
        model_path, os.path.join(os.path.dirname(model_path), "model.json")
    )

    logging.warning("reading model parameters from " + model_path)

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
        model.state_dict() (OrderedDict): the loaded model state_dict
        (bool): Boolean defining whether the model is an LM

    """
    conf_path = os.path.join(os.path.dirname(model_path), "model.json")
    if "rnnlm" in model_path:
        logging.warning("reading model parameters from %s", model_path)

        return torch.load(model_path), True

    idim, odim, args = get_model_conf(model_path, conf_path)

    logging.warning("reading model parameters from " + model_path)

    if hasattr(args, "model_module"):
        model_module = args.model_module
    else:
        model_module = "espnet.nets.pytorch_backend.e2e_asr:E2E"

    model_class = dynamic_import(model_module)
    model = model_class(idim, odim, args)
    torch_load(model_path, model)
    assert (
        isinstance(model, MTInterface)
        or isinstance(model, ASRInterface)
        or isinstance(model, TTSInterface)
    )

    return model.state_dict(), False


def load_trained_modules(idim, odim, args, interface=ASRInterface):
    """Load model encoder or/and decoder modules with ESPNET pre-trained model(s).

    Args:
        idim (int): initial input dimension.
        odim (int): initial output dimension.
        args (Namespace): The initial model arguments.
        interface (Interface): ASRInterface or STInterface or TTSInterface.

    Return:
        model (torch.nn.Module): The model with pretrained modules.

    """

    def print_new_keys(state_dict, modules, model_path):
        logging.warning("loading %s from model: %s", modules, model_path)

        for k in state_dict.keys():
            logging.warning("override %s" % k)

    enc_model_path = args.enc_init
    dec_model_path = args.dec_init
    enc_modules = args.enc_init_mods
    dec_modules = args.dec_init_mods

    model_class = dynamic_import(args.model_module)
    main_model = model_class(idim, odim, args)
    assert isinstance(main_model, interface)

    main_state_dict = main_model.state_dict()

    logging.warning("model(s) found for pre-initialization")
    for model_path, modules in [
        (enc_model_path, enc_modules),
        (dec_model_path, dec_modules),
    ]:
        if model_path is not None:
            if os.path.isfile(model_path):
                model_state_dict, is_lm = get_trained_model_state_dict(model_path)

                modules = filter_modules(model_state_dict, modules)
                if is_lm:
                    partial_state_dict, modules = get_partial_lm_state_dict(
                        model_state_dict, modules
                    )
                    print_new_keys(partial_state_dict, modules, model_path)
                else:
                    partial_state_dict = get_partial_state_dict(
                        model_state_dict, modules
                    )

                    if partial_state_dict:
                        if transfer_verification(
                            main_state_dict, partial_state_dict, modules
                        ):
                            print_new_keys(partial_state_dict, modules, model_path)
                            main_state_dict.update(partial_state_dict)
                        else:
                            logging.warning(
                                f"modules {modules} in model {model_path} "
                                f"don't match your training config",
                            )
            else:
                logging.warning("model was not found : %s", model_path)

    main_model.load_state_dict(main_state_dict)

    return main_model
