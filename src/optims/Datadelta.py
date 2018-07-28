#!/usr/bin/env python
__author__ = 'arenduchintala'
import torch
import numpy as np




class Datadelta(torch.optim.Optimizer):
    """Implements Datadelta algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rho (float, optional): coefficient used for computing a running average
            of squared gradients (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6)
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    __ TODO:write paper get best paper award
    """

    def __init__(self, params,
                 score_func,
                 converter,
                 undo_type='undo_step',
                 lr=1.0,
                 rho=0.9,
                 eps=1e-6,
                 weight_decay=0,
                 valid_batches=None,
                 num_samples = 1,
                 which_batches_to_check=['main', 'aug'],
                 diff_threshold=0.):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= rho <= 1.0:
            raise ValueError("Invalid rho value: {}".format(rho))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        self.valid_batches = valid_batches
        self.score_func = score_func
        self.converter = converter
        self.num_samples = num_samples
        self.diff_threshold = diff_threshold
        self.pv_idx = 0
        assert score_func is not None
        self.which_batches_to_check = which_batches_to_check
        super(Datadelta, self).__init__(params, defaults)
        self.init_pv_scores()
        self.undo_type = undo_type
        self.undo_func = {
                'undo_step': self.undo_step,
                'undo_square_avg': self.undo_square_avg,
                'undo_learning_rate': self.undo_learning_rate
               }

    def undo_step(self):
        #print('undo step')
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                rho, eps = group['rho'], group['eps']
                state = self.state[p]
                state['step'] -= 1
                p.data.add_(-1.0, state['update'])
                square_avg = state['square_avg']
                square_avg.add_(-1.0, state['square_grad']).mul_(1. / rho)
                acc_delta = state['acc_delta']
                acc_delta.add_(-1.0, state['delta']).mul_(1. / rho)
        return True

    def undo_learning_rate(self):
        #print('undo lr')
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                rho, eps = group['rho'], group['eps']
                state = self.state[p]
                square_avg = state['square_avg']
                square_avg.add_(-1.0, state['square_grad']).mul_(1. / rho)
                acc_delta = state['acc_delta']
                acc_delta.add_(-1.0, state['delta']).mul_(1. / rho)
        return True

    def undo_square_avg(self):
        #print('undo sqa')
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                rho, eps = group['rho'], group['eps']
                state = self.state[p]
                square_avg = state['square_avg']
                square_avg.add_(-1.0, state['square_grad']).mul_(1. / rho)
        return True


    def init_pv_scores(self):
        #print('init prev valid scores...')
        self.pv_scores = []
        for b in self.valid_batches:
            #TODO: too dirty hack
            b = self.converter([b])
            x_ctc, x_att, x_acc = self.score_func(b, False)
            self.pv_scores.append(x_acc)
        #print('init score', self.pv_scores)
        return True

    def check(self, batch_type):
        #chaining of batch samples so that we can compare param updates on the same randomly sampled batch 
        return_val = True
        if batch_type in self.which_batches_to_check:
            pv_batch = self.converter([self.valid_batches[self.pv_idx]])
            remaining_valid_batch_idxs = [idx for idx, v in enumerate(self.valid_batches) if idx != self.pv_idx]
            #print('remainging', remaining_valid_batch_idxs)
            nv_idx = np.random.choice(remaining_valid_batch_idxs)
            #print('pv_idx->nv_idx', self.pv_idx, nv_idx)
            pv_score = self.pv_scores[self.pv_idx]
            _, _, valid_score = self.score_func(pv_batch, False)
            diff = valid_score - pv_score
            #print('diff', diff, valid_score, pv_score)
            if diff > self.diff_threshold:  #will apply the grad, so we update the valid score list
                self.pv_scores[self.pv_idx] = valid_score
                nv_batch = self.converter([self.valid_batches[nv_idx]])
                _, _, self.pv_scores[nv_idx] = self.score_func(nv_batch)
                return_val = True
                #print('APPLY -----'+ batch_type)
            else:
                return_val = False
                #print('REJECT -----'+ batch_type)
            self.pv_idx = nv_idx
        else:
            return_val = True
        return return_val


    def _step(self, batch_type, closure=None):
        """Performs a single optimization step.

        Arguments:
            batch_type (string): needed to compute the compute_gate_grad
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for g_idx, group in enumerate(self.param_groups):
            for p_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adadelta does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    state['square_grad'] = torch.zeros_like(p.data)
                    state['acc_delta'] = torch.zeros_like(p.data)
                    state['prev_update'] = torch.zeros_like(p.data)
                    state['prev_acc_delta'] = torch.zeros_like(p.data)

                square_avg, acc_delta = state['square_avg'], state['acc_delta']
                rho, eps = group['rho'], group['eps']
                if p_idx == 0 and g_idx == 0:
                    #print(state['step'])
                    #print(state['square_avg'])
                    #print(state['acc_delta'])
                    pass

                state['step'] += 1

                if group['weight_decay'] != 0:
                    raise NotImplementedError("we dont know how to deal with weight_decay atm")
                    grad = grad.add(group['weight_decay'], p.data)

                #data_rho = rho * self.compute_gate_grad(prev_batch_type)
                #square_avg.mul_(rho).addcmul_(1 - rho, grad, grad)
                square_grad = (1 - rho) * grad * grad
                square_avg.mul_(rho).add_(square_grad)
                state['square_grad'] = square_grad
                std = square_avg.add(eps).sqrt_()
                delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad)
                possible_update = -group['lr'] * delta
                #p.data.add_(-group['lr'], delta)
                #get_scores
                p.data.add_(possible_update)
                state['update'] = possible_update
                #alpha_t = self.compute_gate_grad(batch_type)
                #self.undo_step()
                final_delta = (1 - rho) * delta * delta
                state['delta'] = final_delta
                #acc_delta.mul_(rho).addcmul_(1 - rho, delta, delta)
                acc_delta.mul_(rho).add_(final_delta)
        return loss

    def step(self, batch_type):
        #print('\n===================================================')
        self._step(batch_type)
        if not self.check(batch_type):
            #print('undoing step...')
            self.undo_func[self.undo_type]()
        else:
            #print('keeping step...')
            pass
