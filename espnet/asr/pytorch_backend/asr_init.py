"""Finetuning methods."""

import logging
import os
import re
from collections import OrderedDict

import torch

from espnet.asr.asr_utils import get_model_conf, torch_load
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.mt_interface import MTInterface
from espnet.nets.pytorch_backend.transducer.utils import custom_torch_load
from espnet.nets.tts_interface import TTSInterface
from espnet.utils.dynamic_import import dynamic_import


def freeze_modules(model, modules):
    """Freeze model parameters according to modules list.

    Args:
        model (torch.nn.Module): Main model.
        modules (List): Specified module(s) to freeze.

    Return:
        model (torch.nn.Module) : Updated main model.
        model_params (filter): Filtered model parameters.

    """
    for mod, param in model.named_parameters():
        if any(mod.startswith(m) for m in modules):
            logging.warning(f"Freezing {mod}. It will not be updated during training.")
            param.requires_grad = False

    model_params = filter(lambda x: x.requires_grad, model.parameters())

    return model, model_params


def transfer_verification(model_state_dict, partial_state_dict, modules):
    """Verify tuples (key, shape) for input model modules match specified modules.

    Args:
        model_state_dict (Dict) : Main model state dict.
        partial_state_dict (Dict): Pre-trained model state dict.
        modules (List): Specified module(s) to transfer.

    Return:
        (bool): Whether transfer learning is allowed.

    """
    model_modules = []
    partial_modules = []

    for key_m, value_m in model_state_dict.items():
        if any(key_m.startswith(m) for m in modules):
            model_modules += [(key_m, value_m.shape)]
    model_modules = sorted(model_modules, key=lambda x: (x[0], x[1]))

    for key_p, value_p in partial_state_dict.items():
        if any(key_p.startswith(m) for m in modules):
            partial_modules += [(key_p, value_p.shape)]
    partial_modules = sorted(partial_modules, key=lambda x: (x[0], x[1]))

    module_match = model_modules == partial_modules

    if not module_match:
        logging.error(
            "Some specified modules from the pre-trained model "
            "don't match with the new model modules:"
        )
        logging.error(f"Pre-trained: {set(partial_modules) - set(model_modules)}")
        logging.error(f"New model: {set(model_modules) - set(partial_modules)}")
        exit(1)

    return module_match


def get_partial_state_dict(model_state_dict, modules):
    """Create state dict with specified modules matching input model modules.

    Args:
        model_state_dict (Dict): Pre-trained model state dict.
        modules (Dict): Specified module(s) to transfer.

    Return:
        new_state_dict (Dict): State dict with specified modules weights.

    """
    new_state_dict = OrderedDict()

    for key, value in model_state_dict.items():
        if any(key.startswith(m) for m in modules):
            new_state_dict[key] = value

    return new_state_dict


def get_lm_state_dict(lm_state_dict):
    """Create compatible ASR decoder state dict from LM state dict.

    Args:
        lm_state_dict (Dict): Pre-trained LM state dict.

    Return:
        new_state_dict (Dict): State dict with compatible key names.

    """
    new_state_dict = OrderedDict()

    for key, value in list(lm_state_dict.items()):
        if key == "predictor.embed.weight":
            new_state_dict["dec.embed.weight"] = value
        elif key.startswith("predictor.rnn."):
            _split = key.split(".")

            new_key = "dec.decoder." + _split[2] + "." + _split[3] + "_l0"
            new_state_dict[new_key] = value

    return new_state_dict


def filter_modules(model_state_dict, modules):
    """Filter non-matched modules in model state dict.

    Args:
        model_state_dict (Dict): Pre-trained model state dict.
        modules (List): Specified module(s) to transfer.

    Return:
        new_mods (List): Filtered module list.

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
        logging.error(
            "Specified module(s) don't match or (partially match) "
            f"available modules in model. You specified: {incorrect_mods}."
        )
        logging.error("The existing modules in model are:")
        logging.error(f"{mods_model}")
        exit(1)

    return new_mods


def create_transducer_compatible_state_dict(
    model_state_dict, encoder_type, encoder_units
):
    """Create a compatible transducer model state dict for transfer learning.

    If RNN encoder modules from a non-Transducer model are found in
    the pre-trained model state dict, the corresponding modules keys are
    renamed for compatibility.

    Args:
        model_state_dict (Dict): Pre-trained model state dict
        encoder_type (str): Type of pre-trained encoder.
        encoder_units (int): Number of encoder units in pre-trained model.

    Returns:
        new_state_dict (Dict): Transducer compatible pre-trained model state dict.

    """
    if encoder_type.endswith("p") or not encoder_type.endswith(("lstm", "gru")):
        return model_state_dict

    new_state_dict = OrderedDict()
    rnn_key_name = "birnn" if "b" in encoder_type else "rnn"

    for key, value in list(model_state_dict.items()):
        if any(k in key for k in ["l_last", "nbrnn"]):
            if "nbrnn" in key:
                layer_name = rnn_key_name + re.search("_l([0-9]+)", key).group(1)

                key = re.sub("_l([0-9]+)", "_l0", key.replace("nbrnn", layer_name),)

            if (encoder_units * 2) == value.size(-1):
                value = value[:, :encoder_units] + value[:, encoder_units:]

        new_state_dict[key] = value

    return new_state_dict


def load_trained_model(model_path, training=True):
    """Load the trained model for recognition.

    Args:
        model_path (str): Path to model.***.best
        training (bool): Training mode specification for transducer model.

    Returns:
        model (torch.nn.Module): Trained model.
        train_args (Namespace): Trained model arguments.

    """
    idim, odim, train_args = get_model_conf(
        model_path, os.path.join(os.path.dirname(model_path), "model.json")
    )

    logging.info(f"Reading model parameters from {model_path}")

    if hasattr(train_args, "model_module"):
        model_module = train_args.model_module
    else:
        model_module = "espnet.nets.pytorch_backend.e2e_asr:E2E"

    # CTC Loss is not needed, default to builtin to prevent import errors
    if hasattr(train_args, "ctc_type"):
        train_args.ctc_type = "builtin"

    model_class = dynamic_import(model_module)

    if "transducer" in model_module:
        model = model_class(idim, odim, train_args, training=training)
        custom_torch_load(model_path, model, training=training)
    else:
        model = model_class(idim, odim, train_args)
        torch_load(model_path, model)

    return model, train_args


def get_trained_model_state_dict(model_path, new_is_transducer):
    """Extract the trained model state dict for pre-initialization.

    Args:
        model_path (str): Path to trained model.
        new_is_transducer (bool): Whether the new model is Transducer-based.

    Return:
        (Dict): Trained model state dict.

    """
    logging.info(f"Reading model parameters from {model_path}")

    conf_path = os.path.join(os.path.dirname(model_path), "model.json")

    if "rnnlm" in model_path:
        return get_lm_state_dict(torch.load(model_path))

    idim, odim, args = get_model_conf(model_path, conf_path)

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

    if new_is_transducer and "transducer" not in args.model_module:
        return create_transducer_compatible_state_dict(
            model.state_dict(), args.etype, args.eunits,
        )

    return model.state_dict()


def load_trained_modules(idim, odim, args, interface=ASRInterface):
    """Load ASR/MT/TTS model with pre-trained weights for specified modules.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        args Namespace: Model arguments.
        interface (ASRInterface|MTInterface|TTSInterface): Model interface.

    Return:
        main_model (torch.nn.Module): Model with pre-initialized weights.

    """

    def print_new_keys(state_dict, modules, model_path):
        logging.info(f"Loading {modules} from model: {model_path}")

        for k in state_dict.keys():
            logging.warning(f"Overriding module {k}")

    enc_model_path = args.enc_init
    dec_model_path = args.dec_init
    enc_modules = args.enc_init_mods
    dec_modules = args.dec_init_mods

    model_class = dynamic_import(args.model_module)
    main_model = model_class(idim, odim, args)
    assert isinstance(main_model, interface)

    main_state_dict = main_model.state_dict()
    logging.warning("Model(s) found for pre-initialization.")

    for model_path, modules in [
        (enc_model_path, enc_modules),
        (dec_model_path, dec_modules),
    ]:
        if model_path is not None:
            if os.path.isfile(model_path):
                model_state_dict = get_trained_model_state_dict(
                    model_path, "transducer" in args.model_module
                )
                modules = filter_modules(model_state_dict, modules)

                partial_state_dict = get_partial_state_dict(model_state_dict, modules)

                if partial_state_dict:
                    if transfer_verification(
                        main_state_dict, partial_state_dict, modules
                    ):
                        print_new_keys(partial_state_dict, modules, model_path)
                        main_state_dict.update(partial_state_dict)
            else:
                logging.error(f"Specified model was not found: {model_path}")
                exit(1)

    main_model.load_state_dict(main_state_dict)

    return main_model
