import torch


def to_device(data, device):
    """Change the device of object recursively"""
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    # maybe namedtuple. I don't know the correct way to judge namedtuple.
    elif isinstance(data, tuple) and type(data) is not tuple:
        return type(data)(*[to_device(o, device) for o in data])
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


def force_gatherable(data, device):
    """Change object to gatherable in torch.nn.DataParallel recursively

    The diffrence fron to_device() is changing to torch.Tensor if float or int
    value is found.
    """
    if isinstance(data, dict):
        return {k: force_gatherable(v, device) for k, v in data.items()}
    # maybe namedtuple. I don't know the correct way to judge namedtuple.
    elif isinstance(data, tuple) and type(data) is not tuple:
        return type(data)(*[force_gatherable(o, device) for o in data])
    elif isinstance(data, (list, tuple)):
        return type(data)(force_gatherable(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, float):
        return torch.tensor(data, dtype=torch.float, device=device)
    elif isinstance(data, int):
        return torch.tensor(data, dtype=torch.long, device=device)
    else:
        raise None

