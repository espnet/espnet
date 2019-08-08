import logging
import os
import torch

from collections import OrderedDict

# Note : the following methods are just toy examples for finetuning RNN-T and RNN-T att.
# It's quite inelegant as if but it handle most of the cases.


def load_pretrained_modules(model, rnnt_mode, enc_pretrained, dec_pretrained,
                            enc_mods, dec_mods):
    """Method to update model modules weights from up to two ESPNET pre-trained models.

    Specified models can be either trained with CTC, attention or joint CTC-attention,

    or a language model trained with CE to initialize the decoder part in vanilla RNN-T.

    Args:
        model (torch.nn.Module): initial torch model
        rnnt_mode (int): RNN-transducer mode
        enc_pretrained (str): ESPNET pre-trained model path file to initialize encoder part
        dec_pretrained (str): ESPNET pre-trained model path file to initialize decoder part
    """

    def filter_modules(model, modules):
        new_mods = []
        incorrect_mods = []

        mods_model = list(model.keys())
        for mod in modules:
            if any(key.startswith(mod) for key in mods_model):
                new_mods += [mod]
            else:
                incorrect_mods += [mod]

        if incorrect_mods:
            logging.info("Some specified module(s) for finetuning don\'t "
                         "match (or partially match) available modules."
                         " Disabling the following modules: %s", incorrect_mods)
            logging.info('For information, the existing modules in model are:')
            logging.info('%s', mods_model)

        return new_mods

    def validate_modules(model, prt, modules):
        modules_model = []
        modules_prt = []

        for key_p, value_p in prt.items():
            if any(key_p.startswith(m) for m in modules):
                modules_prt += [(key_p, value_p.shape)]

        for key_m, value_m in model.items():
            if any(key_m.startswith(m) for m in modules):
                modules_model += [(key_m, value_m.shape)]

        len_match = (len(modules_model) == len(modules_prt))
        module_match = (sorted([x for x in modules_model]) ==
                        sorted([x for x in modules_prt]))

        return len_match and module_match

    def get_am_state_dict(model, modules):
        new_state_dict = OrderedDict()

        for key, value in model.items():
            if any(key.startswith(m) for m in modules):
                if 'output' not in key:
                    new_state_dict[key] = value

        return new_state_dict

    def get_lm_state_dict(model, modules):
        new_state_dict = OrderedDict()
        new_modules = []

        for key, value in list(model.items()):
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

    model_state_dict = model.state_dict()

    if rnnt_mode == 0 and dec_pretrained is not None:
        lm_init = True
    else:
        lm_init = False

    for prt_model_path, prt_mods, prt_type in [(enc_pretrained, enc_mods, False),
                                               (dec_pretrained, dec_mods, lm_init)]:
        if prt_model_path is not None:
            if os.path.isfile(prt_model_path):
                prt_model = torch.load(prt_model_path,
                                       map_location=lambda storage, loc: storage)

                prt_mods = filter_modules(prt_model, prt_mods)
                if prt_type:
                    prt_state_dict, prt_mods = get_lm_state_dict(prt_model, prt_mods)
                else:
                    prt_state_dict = get_am_state_dict(prt_model, prt_mods)

                if prt_state_dict:
                    if validate_modules(model_state_dict, prt_state_dict, prt_mods):
                        model_state_dict.update(prt_state_dict)
                    else:
                        logging.info("The model you specified for pre-initialization "
                                     "doesn\'t match your run config. It will be ignored")
                        logging.info('Model path: %s' % prt_model_path)
            else:
                logging.info('The model you specified for pre-initialization was not found.')
                logging.info('Model path: %s' % prt_model_path)

    model.load_state_dict(model_state_dict)

    del model_state_dict

    return model


def freeze_modules(model, modules):
    """Method to freeze specified list of modules.

    Args:
        model (torch.nn.Module): torch model
        modules (str): list of module names to freeze during training

    Returns:
        (boolean): if True, filter the specified modules in the optimizer
    """

    mods_model = [name for name, _ in model.named_parameters()]

    if not any(i in j for j in mods_model for i in modules):
        logging.info("Some module(s) you specified to freeze don\'t "
                     "match (or partially match) available modules.")
        logging.info("Disabling the option.")
        logging.info('For information, the existing modules in model are:')
        logging.info('%s', mods_model)

        return False

    for name, param in model.named_parameters():
        if any(name.startswith(m) for m in modules):
            param.requires_grad = False

    return True
