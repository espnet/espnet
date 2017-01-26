import torch
import warpctc_pytorch as warp_ctc
from torch.autograd import Function
from torch.nn import Module
from torch.nn.modules.loss import _assert_no_grad
from torch.utils.ffi import _wrap_function
from ._warp_ctc import lib as _lib, ffi as _ffi

__all__ = []


def _import_symbols(locals):
    for symbol in dir(_lib):
        fn = getattr(_lib, symbol)
        locals[symbol] = _wrap_function(fn, _ffi)
        __all__.append(symbol)


_import_symbols(locals())


class _CTC(Function):
    def forward(self, acts, labels, act_lens, label_lens):
        """
        acts: Tensor of (seqLength x batch x outputDim) containing output from network
        labels: 1 dimensional Tensor containing all the targets of the batch in one sequence
        act_lens: Tensor of size (batch) containing size of each output sequence from the network
        label_lens: Tensor of (batch) containing label length of each example
        """
        is_cuda = True if acts.is_cuda else False
        loss_func = warp_ctc.gpu_ctc if is_cuda else warp_ctc.cpu_ctc
        grads = torch.zeros(acts.size()).type_as(acts)
        minibatch_size = acts.size(1)
        costs = torch.zeros(minibatch_size)
        loss_func(acts,
                  grads,
                  labels,
                  label_lens,
                  act_lens,
                  minibatch_size,
                  costs)
        self.grads = grads
        self.costs = torch.FloatTensor([costs.sum()])
        if is_cuda:
            self.grads = self.grads.cuda()
            self.costs = self.costs.cuda()
        return self.costs

    def backward(self, grad_output):
        return self.grads, None, None, None


class CTCLoss(Module):
    def __init__(self):
        super(CTCLoss, self).__init__()

    def forward(self, input, target, sizes, label_lens):
        _assert_no_grad(target)
        _assert_no_grad(sizes)
        _assert_no_grad(label_lens)
        return _CTC()(input, target, sizes, label_lens)
