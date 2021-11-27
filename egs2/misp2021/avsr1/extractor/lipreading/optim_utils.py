import math
import torch
import torch.optim as optim


def change_lr_on_optimizer(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class CosineScheduler:
    def __init__(self, lr_ori, epochs):
        self.lr_ori = lr_ori
        self.epochs = epochs

    def adjust_lr(self, optimizer, epoch):
        reduction_ratio = 0.5 * (1 + math.cos(math.pi * epoch / self.epochs))
        change_lr_on_optimizer(optimizer, self.lr_ori*reduction_ratio)


def get_optimizer(args, optim_policies):
    # -- define optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(optim_policies, lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(optim_policies, lr=args.lr, weight_decay=1e-2)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(optim_policies, lr=args.lr, weight_decay=1e-4, momentum=0.9)
    else:
        raise NotImplementedError
    return optimizer
