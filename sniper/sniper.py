import dataclasses
import numpy as np
import collections
import os
import argparse
import time
import logging
from functools import reduce
from typing import Dict, Callable, Tuple, List, Iterable, Union
import torch


class SniperTraining:
    """To start SNIPER training, we have to pass the args listed in train().


    Usage:
    sniper_dir = 'sniper'
    schedule = {0: 20, 20: 10, 50: 0}  # Start with 20% sparsity, then reduce to 10% after 20 epochs and 0% after 50
    model = YourModel(model_args)


    """

    def __init__(self, sniper_dir, device='cuda', logger=logging.root):

        # Sniper init variables
        self.sniper_dir = sniper_dir
        self.device = device
        self.logger = logger

        # Training args
        self.schedule = {}
        self.model_builder = None
        self.snip_module_name = None
        self.batch_iterator = None
        self.get_loss_fn = None
        self.exclude_params = None
        self.train_dtype = None

        # Training state
        self.epoch = 0
        self.current_masks = None
        self.module_to_snip = None
        self.hook_handle = None

    def train(self,
              schedule: Dict[int, float],
              model: torch.nn.Module,
              model_builder: Callable,
              snip_module_name: str,
              batch_iterator: Iterable,
              get_loss_fn: Callable,
              exclude_params: Iterable[str] = ('batch_norm',),
              train_dtype=torch.float32,
              epoch=0,
              ):
        """
        Args:
            sniper_dir: Where all the related files are stored
            schedule: {epoch: sparsity}
            model: Model to snip
            model_builder: Returns a new model
            snip_module_name: Which component of the model to snip. If empty string, whole model will be snipped
            batch_iterator: When iterated over, returns batch that is passed to get_loss_fn
            get_loss_fn: Function to return loss when called as get_loss_fn(model, batch)
            exclude_params: All params containing any of these strings in the full name will not be masked.
            train_dtype: Default float32
            device: Default cuda


        """

        assert 0 in schedule
        self.schedule = schedule
        self.model_builder = model_builder
        self.snip_module_name = snip_module_name
        self.batch_iterator = batch_iterator
        self.get_loss_fn = get_loss_fn
        self.exclude_params = exclude_params
        self.train_dtype = train_dtype
        self.epoch = epoch

        start_sparsity = schedule[0]

        init_values_path = os.path.join(self.sniper_dir, 'init_values.pt')
        total_grads_path = os.path.join(self.sniper_dir, 'total_grads.pt')

        if not os.path.exists(self.sniper_dir):
            os.makedirs(self.sniper_dir)

        if os.path.exists(init_values_path):
            self.logger.info(f'Loading initial model state from {init_values_path}')
            init_values = torch.load(init_values_path, map_location=self.device)
            model.load_state_dict(init_values)
        else:
            self.logger.info(f'Saving initial model state to {init_values_path}')
            init_values = model.state_dict()
            torch.save(init_values, init_values_path)

        sparsities = sorted(schedule.values())
        missing_sparsities = {}
        for sparsity in sparsities:
            if sparsity:  # ignore 0
                masks_path = os.path.join(self.sniper_dir, f'masks_{sparsity}.pt')
                if not os.path.exists(masks_path):
                    missing_sparsities[sparsity] = masks_path

        if missing_sparsities:
            if os.path.exists(total_grads_path):
                self.logger.info(f'Loading gradients from {total_grads_path}')
                total_grads = torch.load(total_grads_path)
            else:
                self.logger.info(f'Computing gradients...')
                total_grads = self.compute_gradients(init_values)
                torch.save(total_grads, total_grads_path)
                self.logger.info(f'Saved gradients to {total_grads_path}')

            for sparsity in sorted(missing_sparsities):
                masks_path = missing_sparsities[sparsity]
                masks = self.create_masks(sparsity, total_grads)
                torch.save(masks, masks_path)
                self.logger.info(f'Saved masks at sparsity {sparsity} to {masks_path}')
                del masks

            del total_grads

        self.logger.info(f'All required sparsities present, loading sparsity {start_sparsity}...')
        start_masks = load_masks(self.sniper_dir, start_sparsity, self.device)
        self.current_masks = start_masks

        self.logger.info(f'Adding mask operation to forward pre-hooks...')
        module_to_snip = get_module_to_snip(model, snip_module_name)
        self.module_to_snip = module_to_snip

        self.hook_handle = module_to_snip.register_forward_pre_hook(hook=get_hook(start_masks))
        # self.hook_handles = register_masks(module_to_snip, start_masks)

    def step(self):
        self.set_epoch(self.epoch + 1)

    def set_epoch(self, epoch):
        self.epoch = epoch
        if epoch in self.schedule:
            new_sparsity = self.schedule[epoch]
            if new_sparsity:
                self.logger.info(f'New sparsity scheduled: {new_sparsity} -- replacing with new mask')
                new_masks = load_masks(self.sniper_dir, new_sparsity, self.device)
                self.current_masks = new_masks
                self.hook_handle.remove()
                self.hook_handle = self.module_to_snip.register_forward_pre_hook(hook=get_hook(new_masks))
                # update_masks(self.module_to_snip, new_masks)
            else:
                self.logger.info(f'New sparsity is 0 -- removing mask')
                self.current_masks = None
                self.hook_handle.remove()
                # for hook_handle in self.hook_handles:
                #     hook_handle.remove()
                self.hook_handle = None

    def create_masks(self, sparsity: float, total_grads: Dict[str, torch.Tensor]):
        flattened_grads = torch.cat([total_grad.view(-1) for total_grad in total_grads.values()])
        threshold = torch.kthvalue(flattened_grads, int(sparsity / 100. * len(flattened_grads))).values.item()
        masks = {}
        for full_name, total_grad in total_grads.items():
            mask = total_grad > threshold
            masks[full_name] = mask
            self.logger.info(f'{full_name}: {mask.numel()} -> {mask.sum().item()}')

        return masks

    def compute_gradients(self, init_values: collections.OrderedDict):

        start = time.time()

        model = self.load_model(init_values)
        module_to_snip = get_module_to_snip(model, self.snip_module_name)

        param_sizes = {full_name: p.numel() for full_name, p in module_to_snip.named_parameters() if p.requires_grad}
        self.logger.info('Total trainable params: {}'.format(sum(param_sizes.values())))
        params_to_prune = [full_name for full_name, _ in module_to_snip.named_parameters()
                           if will_prune(param_name=full_name, exclude_params=self.exclude_params)]
        self.logger.info('Total params eligible to prune: {}'.format(
            sum(param_sizes[full_name] for full_name in params_to_prune)))

        masks = mask_params(module_to_snip, params_to_prune, self.device)
        total_grads = [torch.zeros_like(mask).to(device=self.device) for mask in masks]

        # each batch in batch_iterator consists of {input_name: tensor} or {input_name: id}
        for i, batch in enumerate(self.batch_iterator):
            batch = to_device(batch, device=self.device)
            loss = self.get_loss_fn(model, batch)
            grads = torch.autograd.grad(loss, masks)
            total_grads = [total_grad + grad for total_grad, grad in zip(total_grads, grads)]
            # Recreate the whole model to avoid backwarding through graph 2nd time
            model = self.load_model(init_values)
            module_to_snip = get_module_to_snip(model, self.snip_module_name)
            masks = mask_params(module_to_snip, params_to_prune, self.device)
            module_to_snip.zero_grad()

        total_grads = {full_name: total_grad.abs() for full_name, total_grad in zip(params_to_prune, total_grads)}

        time_taken = time.time() - start
        self.logger.info(f'SNIP time: {time_taken}')
        self.logger.info(f'SNIP time/batch: {time_taken / (i + 1)}')
        return total_grads

    def load_model(self, init_values):
        log_level = self.logger.getEffectiveLevel()
        self.logger.setLevel(logging.WARNING)
        model = self.model_builder()
        model = model.to(dtype=self.train_dtype, device=self.device)
        model.load_state_dict(init_values)
        self.logger.setLevel(log_level)
        return model


def load_masks(sniper_dir, sparsity, device):
    masks_path = os.path.join(sniper_dir, f'masks_{sparsity}.pt')
    return torch.load(masks_path, map_location=device)


# Solution by albanD @ https://discuss.pytorch.org/t/use-forward-pre-hook-to-modify-nn-module-parameters/108498
def get_hook(masks):
    def hook(module_to_snip, inp):
        for full_name, mask in masks.items():
            m, n = full_name.rsplit('.', 1)
            last_module = get_module_by_name(module_to_snip, m)
            param = getattr(last_module, n)
            param.data = param.data * mask.to(param.dtype)

    return hook


def register_masks(module_to_snip, masks):
    hook_handles = []
    for full_name, mask in masks.items():
        m, n = full_name.rsplit('.', 1)
        last_module = get_module_by_name(module_to_snip, m)
        last_module.register_buffer(n + '_mask', mask, persistent=False)
        hook_handle = last_module.register_forward_pre_hook(hook=apply_mask_hook)
        hook_handles.append(hook_handle)
    return hook_handles


def update_masks(module_to_snip, masks):
    for full_name, mask in masks.items():
        m, n = full_name.rsplit('.', 1)
        last_module = get_module_by_name(module_to_snip, m)
        mask_buffer = getattr(last_module, n + '_mask')
        mask_buffer.data = mask.data


def apply_mask_hook(last_module: torch.nn.Module, inp):  # inp is needed to match hook signature
    param_names = last_module.named_parameters()
    for n in param_names:
        param = getattr(last_module, n)
        mask_buffer = getattr(last_module, n + '_mask')
        param.data = param.data * mask_buffer.to(param.dtype)


def mask_params(module_to_snip, params_to_prune, device) -> Dict[str, torch.Tensor]:
    # This just multiplies all params by 1 and reinserts them into the module
    # Necessary to avoid weird leaf errors when doing autograd
    fmn = [(full_name, *full_name.rsplit('.', 1)) for full_name in params_to_prune]
    fmn = [(full_name, get_module_by_name(module_to_snip, m), n) for full_name, m, n in fmn]
    masks = []
    for full_name, module, name in fmn:
        param = getattr(module, name)
        mask = torch.ones_like(param, requires_grad=True).to(device=device)
        param_masked = param * mask
        with torch.no_grad():
            delattr(module, name)
            setattr(module, name, param_masked)
        masks.append(mask)
        assert mask.is_leaf and not getattr(module, name).is_leaf
    return masks


def get_module_to_snip(model, snip_module_name) -> torch.nn.Module:
    try:
        module_to_snip = get_module_by_name(model, snip_module_name)
    except AttributeError:
        module_to_snip = model
    return module_to_snip


def will_prune(param_name, exclude_params):
    for exclude_param in exclude_params:
        if exclude_param in param_name:
            return False
    return True


def get_module_by_name(model, access_string) -> torch.nn.Module:
    names = access_string.split(sep='.')
    return reduce(getattr, names, model)


def get_param_by_name(module, full_name) -> torch.Tensor:
    m, n = full_name.rsplit('.', 1)
    last_module = get_module_by_name(module, m)
    try:
        return getattr(last_module, n)
    except AttributeError:
        return None


def log_nonzeros_count(module_to_snip, logger=logging.root):
    nonzeros = 0
    numels = 0
    for param in module_to_snip.parameters():
        nonzeros += torch.count_nonzero(param).item()
        numels += param.numel()
    sparsity = 100.0 * (1 - nonzeros / numels)
    logger.info(f'Module has {nonzeros} / {numels} parameters (sparsity {sparsity:.2f}%)')


def to_device(data, device=None, dtype=None, non_blocking=False, copy=False):
    """Change the device of object recursively. Copied from espnet/espnet2/torch_utils/device_funcs.py"""
    if isinstance(data, dict):
        return {
            k: to_device(v, device, dtype, non_blocking, copy) for k, v in data.items()
        }
    elif dataclasses.is_dataclass(data) and not isinstance(data, type):
        return type(data)(
            *[
                to_device(v, device, dtype, non_blocking, copy)
                for v in dataclasses.astuple(data)
            ]
        )
    elif isinstance(data, tuple) and type(data) is not tuple:
        return type(data)(
            *[to_device(o, device, dtype, non_blocking, copy) for o in data]
        )
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device, dtype, non_blocking, copy) for v in data)
    elif isinstance(data, np.ndarray):
        return to_device(torch.from_numpy(data), device, dtype, non_blocking, copy)
    elif isinstance(data, torch.Tensor):
        return data.to(device, dtype, non_blocking, copy)
    else:
        return data
