import random
from pathlib import Path
import dataclasses
import numpy as np
import collections
import os
import time
import logging
from functools import reduce
from typing import Dict, Callable, Iterable
import torch


class SniperTraining:

    def __init__(self, sniper_dir, device='cuda', logger=logging.root):

        # Sniper init variables
        self.sniper_dir = Path(sniper_dir)
        self.device = device
        self.logger = logger

        # Training args
        self.schedule = {}
        self.model_builder = None
        self.snip_module_name = ''
        self.batch_iterator = None
        self.get_loss_fn = None
        self.max_lr_scaling = 10.0
        self.max_param_sparsity = 100.0
        self.exclude_params = None
        self.restore_init_values = False
        self.train_dtype = None
        self.resume = False
        self.optim_lr = 1.0
        self.track_all_params_lr = False

        # Training state
        self.epoch = 0
        self.current_sparsity = 0
        self.current_masks = None
        self.log_sparsity = False
        self.module_to_snip = None
        self.forward_hook = None
        self.param_groups = None
        self.track_pg_index = -1
        self.optimizers = None
        self.schedulers = None


    def train(self,
              schedule: Dict[int, float],
              model: torch.nn.Module,
              model_builder: Callable,
              snip_module_name: str,
              batch_iterator: Iterable,
              get_loss_fn: Callable,
              scale_lr_by_param: bool = True,
              max_lr_scaling: float = 10.0,
              max_param_sparsity: float = 100.0,
              exclude_params: Iterable[str] = ('embed', 'norm',),
              restore_init_values: bool = True,
              train_dtype=torch.float32,
              resume=False,
              optim_lr=1.0,
              track_all_params_lr=False,
              ):
        """

        Args:
            schedule:
                Dict of epoch : sparsity level (in %)
            model:
                The model to be trained and pruned
            model_builder:
                When called, should return an exact copy of model (needed to compute gradients repeatedly)
            snip_module_name:
                Submodule to prune (example: "tts.encoder"). Use empty string to prune whole model.
            batch_iterator:
                Generates batches to `forward()` through the model and compute loss.
            get_loss_fn:
                When `get_loss_fn(model, batch)` is called, should return a differentiable loss function.
            scale_lr_by_param:
                Whether or not to scale learning rate according to each parameter's sparsity. Otherwise, `optim_lr`
                will be used.
            max_lr_scaling:
                When using per-parameter learning rate, sets a maximum LR to prevent gradient explosion.
            max_param_sparsity:
                When computing masks, limits the parameter sparsity to this value to prevent bottlenecks.
            exclude_params:
                Parameters containing these strings will not be pruned.
            restore_init_values:
                If True, when sparsity is reduced, newly activated weights take their initial values.
                If False, newly activated weights will be set to zero.
            train_dtype:
                Should match model's dtype.
            resume:
                If set to True, you MUST call `resume_from()` to restore the epoch count, optimizers and schedulers.
            optim_lr:
                Default optimizer learning rate.
            track_all_params_lr:
                Whether or not to track the learning rate for all parameters if `scale_lr_by_param` is True. This module
                does not use it directly, but you may want to report the LRs in Tensorboard.

        Returns:

        """

        assert 0 in schedule
        if schedule[0] == 0:
            assert len(schedule) == 1
        self.schedule = schedule
        start_sparsity = schedule[0]
        self.current_sparsity = start_sparsity
        self.model_builder = model_builder
        self.snip_module_name = snip_module_name
        self.batch_iterator = batch_iterator
        self.get_loss_fn = get_loss_fn
        self.max_lr_scaling = max_lr_scaling
        self.max_param_sparsity = max_param_sparsity
        self.exclude_params = exclude_params
        self.restore_init_values = restore_init_values
        self.train_dtype = train_dtype
        self.resume = resume
        self.optim_lr = optim_lr
        self.track_all_params_lr = track_all_params_lr

        # self.grad_scaling = 100.0 / (100 - start_sparsity)
        # self.logger.info(f'Gradient scaling is {self.grad_scaling}...')

        self.module_to_snip = get_module_to_snip(model, snip_module_name)

        init_values_path = self.sniper_dir / 'init_values.pt'
        total_grads_path = self.sniper_dir / 'total_grads.pt'

        if not self.sniper_dir.exists():
            os.makedirs(self.sniper_dir)

        if init_values_path.exists():
            self.logger.info(f'Loading initial model state from {init_values_path}')
            init_values = torch.load(init_values_path, map_location=self.device)
            if not resume:
                model.load_state_dict(init_values)
        else:
            self.logger.info(f'Saving initial model state to {init_values_path}')
            init_values = model.state_dict()
            torch.save(init_values, init_values_path)

        sparsities = sorted(schedule.values())
        missing_sparsities = {}
        for sparsity in sparsities:
            if sparsity:  # ignore 0
                max_suffix = '' if max_param_sparsity == 100.0 else f'_max{max_param_sparsity}'
                masks_path = self.sniper_dir / f'masks_{sparsity}{max_suffix}.pt'
                if not os.path.exists(masks_path):
                    missing_sparsities[sparsity] = masks_path

        if missing_sparsities:
            if total_grads_path.exists():
                self.logger.info(f'Loading gradients from {total_grads_path}')
                total_grads = torch.load(total_grads_path)
            else:
                self.logger.info(f'Computing gradients...')
                total_grads = self.compute_gradients(init_values)
                torch.save(total_grads, total_grads_path)
                self.logger.info(f'Saved gradients to {total_grads_path}')

            for sparsity in sorted(missing_sparsities):
                masks_path = missing_sparsities[sparsity]
                masks = self.create_masks(sparsity, total_grads, max_param_sparsity)
                torch.save(masks, masks_path)
                self.logger.info(f'Saved masks at sparsity {sparsity} to {masks_path}')
                del masks

            del total_grads

        if start_sparsity:
            param_groups = []
            if resume or not scale_lr_by_param:
                for model_full_name, param in model.named_parameters():
                    param_groups.append({'name': model_full_name, 'params': [param], 'lr': optim_lr})
            else:
                self.logger.info(f'All required sparsities present, loading sparsity {start_sparsity}...')
                start_masks = self.load_masks(start_sparsity)
                self.current_masks = start_masks

                self.logger.info(f'Adding mask operation to forward hook...')
                self.forward_hook = self.module_to_snip.register_forward_pre_hook(hook=get_forward_hook(start_masks))
                # self.forward_hooks = register_masks(module_to_snip, start_masks)
                # log_nonzeros_count(self.module_to_snip, self.logger)

                param_groups = []  # full_name: {'name': str, 'params': [Tensor], 'lr': float}
                self.logger.info('Creating optimizer learning rates')

                cutoff = len(self.snip_module_name) + 1 if self.snip_module_name else 0
                for model_full_name, param in model.named_parameters():
                    if model_full_name.startswith(self.snip_module_name):
                        full_name = model_full_name[cutoff:]
                        if full_name in start_masks:
                            mask = start_masks[full_name]
                            density = mask.sum().item() / torch.numel(mask)
                            if density:
                                lr = optim_lr * min(1.0 / density, max_lr_scaling)
                            else:
                                lr = optim_lr
                        else:
                            lr = optim_lr
                    else:
                        lr = optim_lr
                    param_groups.append({'name': model_full_name, 'params': [param], 'lr': lr})
            self.param_groups = param_groups

            if not track_all_params_lr:
                for i, pg in enumerate(param_groups):
                    if pg['lr'] == optim_lr:
                        self.track_pg_index = i
                        break

    def step(self):
        self.epoch += 1
        if self.epoch in self.schedule:
            new_sparsity = self.schedule[self.epoch]
            self.update_hooks(new_sparsity)
            if new_sparsity:
                self.update_lrs()
            else:
                self.reset_lrs()
            if self.restore_init_values:
                self.restore_init()
            else:
                self.log_sparsity = True
        if self.log_sparsity:
            self.log_nonzeros_count()

    def resume_from(self, epoch, optimizers, schedulers):
        self.epoch = epoch
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.logger.info(f'Resuming from epoch {epoch}')
        resume_sparsity = get_sparsity(self.schedule, epoch)
        self.update_hooks(resume_sparsity)
        # No need to update learning rates here as they should be loaded when resuming checkpoint

    def update_hooks(self, new_sparsity):
        self.current_sparsity = new_sparsity
        if self.forward_hook is not None:
            self.forward_hook.remove()
            # for forward_hook in self.forward_hooks:
            #     forward_hook.remove()
        if new_sparsity:
            self.logger.info(f'New sparsity scheduled: {new_sparsity} -- replacing with new mask')
            new_masks = self.load_masks(new_sparsity)
            self.current_masks = new_masks
            self.forward_hook = self.module_to_snip.register_forward_pre_hook(hook=get_forward_hook(new_masks))
            # update_masks(self.module_to_snip, new_masks)

        else:
            self.logger.info(f'New sparsity is 0 -- removing mask')
            self.current_masks = None
            self.forward_hook = None

    def update_lrs(self):
        if self.optimizers is not None and self.schedulers is not None:
            self.logger.info('Setting optimizer and scheduler learning rates')
            cutoff = len(self.snip_module_name) + 1 if self.snip_module_name else 0
            for optim, scheduler in zip(self.optimizers, self.schedulers):
                for param_group in optim.param_groups:
                    model_full_name = param_group['name']
                    full_name = model_full_name[cutoff:]
                    if full_name in self.current_masks:
                        mask = self.current_masks[full_name]
                        density = mask.sum().item() / torch.numel(mask)
                        if density:
                            new_lr = self.optim_lr * min(1.0 / density, self.max_lr_scaling)
                            param_group['lr'] = new_lr
                scheduler.base_lrs = [group['lr'] for group in optim.param_groups]

    def reset_lrs(self):
        if self.optimizers is not None and self.schedulers is not None:
            self.logger.info('Restoring original optimizer and scheduler learning rates')
            for optim, scheduler in zip(self.optimizers, self.schedulers):
                for param_group in optim.param_groups:
                    param_group['lr'] = self.optim_lr
                scheduler.base_lrs = [self.optim_lr] * len(optim.param_groups)

    def create_masks(self, sparsity: float, total_grads: Dict[str, torch.Tensor], max_param_sparsity: float = 100.0):
        flattened_grads = torch.cat([total_grad.view(-1) for total_grad in total_grads.values()])
        threshold = torch.kthvalue(flattened_grads, int(sparsity / 100. * len(flattened_grads))).values.item()
        max_sparsity = max_param_sparsity / 100.0
        masks = {}
        for full_name, total_grad in total_grads.items():
            mask = total_grad > threshold
            nonzero = mask.sum().item()
            numel = mask.numel()
            if 1 - nonzero / numel > max_sparsity:
                grad = total_grad.view(-1)
                param_threshold = torch.kthvalue(grad, 1 + int(max_sparsity * numel)).values.item()
                mask = total_grad > param_threshold
                nonzero = mask.sum().item()
                if nonzero == 0:  # randomly fill the mask to max_sparsity
                    num_false = int(max_sparsity * numel)
                    nonzero = numel - num_false
                    bools = [False] * num_false + [True] * nonzero
                    random.Random(0).shuffle(bools)
                    mask = torch.BoolTensor(bools).reshape_as(mask)
            masks[full_name] = mask
            self.logger.info(f'{full_name}: {numel} -> {nonzero}')

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

    def log_nonzeros_count(self):
        nonzeros = 0
        numels = 0
        for param in self.module_to_snip.parameters():
            nonzeros += torch.count_nonzero(param).item()
            numels += param.numel()
        sparsity = 100.0 * (1 - nonzeros / numels)
        self.logger.info(
            f'Module has {nonzeros} / {numels} parameters (sparsity {sparsity:.2f}%) at epoch {self.epoch}')

    def restore_init(self):
        init_values_path = self.sniper_dir / 'init_values.pt'
        init_values = torch.load(init_values_path, map_location=self.device)
        # The keys in init_values are fully qualified names for the whole model,
        # whereas in masks, they are the qualified names for module_to_snip
        if self.current_masks is None:
            # Remove snip_module_name from the fully qualified name
            # e.g. if module_to_snip = model.tts, we need to make 'tts.encoder.weight' -> 'encoder.weight'
            drop_len = len(self.snip_module_name) + 1 if self.snip_module_name else 0
            for full_name, param_init_values in init_values.items():
                if full_name.startswith(self.snip_module_name):
                    m, n = full_name.rsplit('.', 1)
                    m = m[drop_len:]
                    last_module = get_module_by_name(self.module_to_snip, m)
                    param = getattr(last_module, n)
                    values_to_update = (param.data == 0.0)
                    param.data = param_init_values.data * values_to_update.to(param.dtype) + param.data
        else:
            # Prepend snip_module_name to the qualified name
            # e.g. if module_to_snip = model.tts, we need to make 'encoder.weight' -> 'tts.encoder.weight'
            prefix = self.snip_module_name + '.' if self.snip_module_name else ''
            for full_name, mask in self.current_masks.items():
                param_init_values = init_values[prefix + full_name]
                m, n = full_name.rsplit('.', 1)
                last_module = get_module_by_name(self.module_to_snip, m)
                param = getattr(last_module, n)
                values_to_update = torch.logical_and(param.data == 0.0, mask)
                param.data = param_init_values.data * values_to_update.to(param.dtype) + param.data

    def load_masks(self, sparsity):
        max_suffix = '' if self.max_param_sparsity == 100.0 else f'_max{self.max_param_sparsity}'
        masks_path = self.sniper_dir / f'masks_{sparsity}{max_suffix}.pt'
        masks = torch.load(masks_path, map_location=self.device)
        for full_name in list(masks.keys()):
            for exclude_param in self.exclude_params:
                if exclude_param in full_name:
                    del masks[full_name]
                    break
        return masks


def get_sparsity(schedule, epoch):
    schedule_epochs = sorted(schedule.keys())
    i = 0
    while i < len(schedule_epochs) and epoch >= schedule_epochs[i]:
        i += 1
    match_epoch = schedule_epochs[i-1]
    return schedule[match_epoch]


# Solution by albanD @ https://discuss.pytorch.org/t/use-forward-pre-hook-to-modify-nn-module-parameters/108498
def get_forward_hook(masks):
    def hook(module_to_snip, inp):
        for full_name, mask in masks.items():
            m, n = full_name.rsplit('.', 1)
            last_module = get_module_by_name(module_to_snip, m)
            param = getattr(last_module, n)
            param.data = param.data * mask.to(param.dtype)

    return hook


def get_backward_hook(masks):
    def hook(module_to_snip, grad_input, grad_output):
        for full_name, mask in masks.items():
            m, n = full_name.rsplit('.', 1)
            last_module = get_module_by_name(module_to_snip, m)
            param = getattr(last_module, n)
            param.grad = param.grad * mask.to(param.dtype)
    return hook


def register_forward_hooks(module_to_snip, masks):
    forward_hooks = []
    for full_name, mask in masks.items():
        m, n = full_name.rsplit('.', 1)
        last_module = get_module_by_name(module_to_snip, m)
        last_module.register_buffer(n + '_mask', mask, persistent=False)
        forward_hook = last_module.register_forward_pre_hook(hook=apply_weights_hook)
        forward_hooks.append(forward_hook)
    return forward_hooks


def register_backward_hooks(module_to_snip, masks, grad_scaling):
    backward_hooks = []
    for full_name, mask in masks.items():
        m, n = full_name.rsplit('.', 1)
        last_module = get_module_by_name(module_to_snip, m)
        param = getattr(last_module, n)
        backward_hook = param.register_hook(
            lambda grad, grad_mask=mask, scaling=grad_scaling: grad.mul_(grad_mask).mul_(scaling))
        backward_hooks.append(backward_hook)
    return backward_hooks


def update_masks(module_to_snip, masks):
    for full_name, mask in masks.items():
        m, n = full_name.rsplit('.', 1)
        last_module = get_module_by_name(module_to_snip, m)
        mask_buffer = getattr(last_module, n + '_mask')
        mask_buffer.data = mask.data


def apply_weights_hook(last_module: torch.nn.Module, inp):  # inp is needed to match hook signature
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
