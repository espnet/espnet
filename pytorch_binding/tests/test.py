import torch
from torch.autograd import Variable

from warpctc_pytorch import CTCLoss

ctc_loss = CTCLoss()


def simple_test():
    probs = torch.FloatTensor([[[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]]]).transpose(0, 1).contiguous()
    labels = Variable(torch.IntTensor([1, 2]))
    label_sizes = Variable(torch.IntTensor([2]))
    sizes = Variable(torch.IntTensor([2]))
    probs = Variable(probs, requires_grad=True)
    cost = ctc_loss(probs, labels, sizes, label_sizes)
    cost.backward()
    print('CPU_cost: %f' % cost.data[0])
    probs = Variable(probs.data.cuda(), requires_grad=True)
    cost = ctc_loss(probs, labels, sizes, label_sizes)
    cost.backward()
    print('GPU_cost: %f' % cost.data[0])
    grads = probs.grad
    print(grads.view(grads.size(0) * grads.size(1), grads.size(2)))


def medium_test(multiplier):
    probs = torch.FloatTensor([
        [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
        [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]
    ]).contiguous() * multiplier

    labels = Variable(torch.IntTensor([1, 2, 1, 2]))
    label_sizes = Variable(torch.IntTensor([2, 2]))
    sizes = Variable(torch.IntTensor([2, 2]))
    probs = Variable(probs, requires_grad=True)
    cost = ctc_loss(probs, labels, sizes, label_sizes)
    cost.backward()
    print('CPU_cost: %f' % cost.data[0])
    probs = Variable(probs.data.cuda(), requires_grad=True)
    cost = ctc_loss(probs, labels, sizes, label_sizes)
    cost.backward()
    print('GPU_cost: %f' % cost.data[0])
    grads = probs.grad
    print(grads.view(grads.size(0) * grads.size(1), grads.size(2)))


def empty_label_test():
    probs = torch.FloatTensor([
        [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
        [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]
    ]).contiguous()

    labels = Variable(torch.IntTensor([1, 2]))
    label_sizes = Variable(torch.IntTensor([2, 0]))
    sizes = Variable(torch.IntTensor([2, 2]))
    probs = Variable(probs, requires_grad=True)
    cost = ctc_loss(probs, labels, sizes, label_sizes)
    cost.backward()
    print('CPU_cost: %f' % cost.data[0])
    probs = Variable(probs.data.cuda(), requires_grad=True)
    cost = ctc_loss(probs, labels, sizes, label_sizes)
    cost.backward()
    print('GPU_cost: %f' % cost.data[0])
    grads = probs.grad
    print(grads.view(grads.size(0) * grads.size(1), grads.size(2)))

simple_test()
medium_test(1.0)
print("Stability test")
medium_test(200.0)  # test SM stability if compiled with USE_NSM this will not have nans
print("Empty label test")
empty_label_test()
