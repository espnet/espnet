import torch
import warpctc_pytorch as warp_ctc


def simple_test():
    probs = torch.FloatTensor([[[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]]]).transpose(0, 1).contiguous()
    grads = torch.zeros(probs.size())
    labels = torch.IntTensor([1, 2])
    label_sizes = torch.IntTensor([2])
    sizes = torch.IntTensor(probs.size(1)).fill_(probs.size(0))
    minibatch_size = probs.size(1)
    costs = torch.zeros(minibatch_size)
    warp_ctc.cpu_ctc(probs,
                     grads,
                     labels,
                     label_sizes,
                     sizes,
                     minibatch_size,
                     costs)
    print('CPU_cost: %f' % costs.sum())
    probs = probs.clone().cuda()
    grads = torch.zeros(probs.size()).cuda()
    costs = torch.zeros(minibatch_size)
    warp_ctc.gpu_ctc(probs,
                     grads,
                     labels,
                     label_sizes,
                     sizes,
                     minibatch_size,
                     costs)
    print('GPU_cost: %f' % costs.sum())
    print(grads.view(grads.size(0) * grads.size(1), grads.size(2)))


def medium_test(multiplier):
    probs = torch.FloatTensor([
        [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
        [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]
    ]).contiguous() * multiplier

    grads = torch.zeros(probs.size())
    labels = torch.IntTensor([1, 2, 1, 2])
    label_sizes = torch.IntTensor([2, 2])
    sizes = torch.IntTensor([2, 2])
    minibatch_size = probs.size(1)
    costs = torch.zeros(minibatch_size)
    warp_ctc.cpu_ctc(probs,
                     grads,
                     labels,
                     label_sizes,
                     sizes,
                     minibatch_size,
                     costs)
    print('CPU_cost: %f' % costs.sum())
    probs = probs.clone().cuda()
    grads = torch.zeros(probs.size()).cuda()
    costs = torch.zeros(minibatch_size)
    warp_ctc.gpu_ctc(probs,
                     grads,
                     labels,
                     label_sizes,
                     sizes,
                     minibatch_size,
                     costs)
    print('GPU_cost: %f' % costs.sum())
    print(grads.view(grads.size(0) * grads.size(1), grads.size(2)))


def empty_label_test():
    probs = torch.FloatTensor([
        [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
        [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]
    ]).contiguous()

    grads = torch.zeros(probs.size())
    labels = torch.IntTensor([1, 2])
    label_sizes = torch.IntTensor([2, 0])
    sizes = torch.IntTensor([2, 2])
    minibatch_size = probs.size(1)
    costs = torch.zeros(minibatch_size)
    warp_ctc.cpu_ctc(probs,
                     grads,
                     labels,
                     label_sizes,
                     sizes,
                     minibatch_size,
                     costs)
    print('CPU_cost: %f' % costs.sum())
    probs = probs.clone().cuda()
    grads = torch.zeros(probs.size()).cuda()
    costs = torch.zeros(minibatch_size)
    warp_ctc.gpu_ctc(probs,
                     grads,
                     labels,
                     label_sizes,
                     sizes,
                     minibatch_size,
                     costs)
    print('GPU_cost: %f' % costs.sum())
    print(grads.view(grads.size(0) * grads.size(1), grads.size(2)))

simple_test()
medium_test(1.0)
print("Stability test")
medium_test(200.0)  # test SM stability if compiled with USE_NSM this will not have nans
print("Empty label test")
empty_label_test()
