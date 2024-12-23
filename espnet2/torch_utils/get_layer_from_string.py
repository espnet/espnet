import difflib

import torch


def get_layer(l_name, library=torch.nn):
    """
        Return layer object handler from a specified library, such as torch.nn.

    This function searches for a layer in the given library based on its name.
    If a matching layer is found, it returns the corresponding layer object handler.

    Example:
        If `l_name` is "elu", the function will return `torch.nn.ELU`.

    Args:
        l_name (str): Case insensitive name for the layer in the library (e.g. 'elu').
        library (module): Name of the library/module to search for the object handler
                          with `l_name`, e.g. `torch.nn`.

    Returns:
        layer_handler (object): Handler for the requested layer (e.g. `torch.nn.ELU`).

    Raises:
        NotImplementedError: If no matching layer is found or if multiple matches
        are found for the specified layer name.
    """

    all_torch_layers = [x for x in dir(torch.nn)]
    match = [x for x in all_torch_layers if l_name.lower() == x.lower()]
    if len(match) == 0:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers]
        )
        raise NotImplementedError(
            "Layer with name {} not found in {}.\n Closest matches: {}".format(
                l_name, str(library), close_matches
            )
        )
    elif len(match) > 1:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers]
        )
        raise NotImplementedError(
            "Multiple matchs for layer with name {} not found in {}.\n "
            "All matches: {}".format(l_name, str(library), close_matches)
        )
    else:
        # valid
        layer_handler = getattr(library, match[0])
        return layer_handler
