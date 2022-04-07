import torch
import difflib


def get_layer(act_name, library=torch.nn):
    """
    Return layer object handler from library e.g. from torch.nn
    E.g. if act_name=="elu", returns torch.nn.ELU

        Args:
            act_name: Case insensitive name for layer in library (e.g. torch.nn).
            library: Name of library/module where to search for object handler with act_name.

    """
    all_torch_layers = [x for x in dir(torch.nn)]
    match = [x for x in all_torch_layers if act_name.lower() == x.lower()]
    if len(match) == 0:
        close_matches = difflib.get_close_matches(
            act_name, [x.lower() for x in all_torch_layers]
        )
        raise NotImplementedError(
            "Layer with name {} not found in {}.\n Closest matches: {}".format(
                act_name, str(library), close_matches
            )
        )
    elif len(match) > 1:
        close_matches = difflib.get_close_matches(
            act_name, [x.lower() for x in all_torch_layers]
        )
        raise NotImplementedError(
            "Multiple matchse for layer with name {} not found in {}.\n All matches: {}".format(
                act_name, str(library), close_matches
            )
        )
    else:
        # valid
        activation_handler = getattr(library, match[0])
        return activation_handler
