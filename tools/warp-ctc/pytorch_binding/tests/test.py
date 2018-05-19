"""
Tests are based on the torch7 bindings for warp-ctc. Reference numbers are also obtained from the tests.
"""
import unittest

import torch
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

ctc_loss = CTCLoss()
places = 5

use_cuda = torch.cuda.is_available()

def run_grads(label_sizes, labels, probs, sizes):
    probs = Variable(probs, requires_grad=True)
    cost = ctc_loss(probs, labels, sizes, label_sizes)
    cost.backward()
    cpu_cost = cost.data[0]
    if use_cuda:
        probs = Variable(probs.data.cuda(), requires_grad=True)
        cost = ctc_loss(probs, labels, sizes, label_sizes)
        cost.backward()
        gpu_cost = cost.data[0]
    else:
        gpu_cost = cpu_cost
    grads = probs.grad
    print(grads.view(grads.size(0) * grads.size(1), grads.size(2)))
    return cpu_cost, gpu_cost


class TestCases(unittest.TestCase):
    def test_simple(self):
        probs = torch.FloatTensor([[[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]]]).transpose(0, 1).contiguous()
        labels = Variable(torch.IntTensor([1, 2]))
        label_sizes = Variable(torch.IntTensor([2]))
        sizes = Variable(torch.IntTensor([2]))
        cpu_cost, gpu_cost = run_grads(label_sizes, labels, probs, sizes)
        expected_cost = 2.4628584384918
        self.assertEqual(cpu_cost, gpu_cost)
        self.assertAlmostEqual(cpu_cost, expected_cost, places)

    def test_medium(self):
        probs = torch.FloatTensor([
            [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
            [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]
        ]).contiguous()

        labels = Variable(torch.IntTensor([1, 2, 1, 2]))
        label_sizes = Variable(torch.IntTensor([2, 2]))
        sizes = Variable(torch.IntTensor([2, 2]))
        cpu_cost, gpu_cost = run_grads(label_sizes, labels, probs, sizes)
        expected_cost = 6.0165174007416
        self.assertEqual(cpu_cost, gpu_cost)
        self.assertAlmostEqual(cpu_cost, expected_cost, places)

    def test_medium_stability(self):
        multiplier = 200
        probs = torch.FloatTensor([
            [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
            [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]
        ]).contiguous() * multiplier

        labels = Variable(torch.IntTensor([1, 2, 1, 2]))
        label_sizes = Variable(torch.IntTensor([2, 2]))
        sizes = Variable(torch.IntTensor([2, 2]))
        cpu_cost, gpu_cost = run_grads(label_sizes, labels, probs, sizes)
        expected_cost = 199.96618652344
        self.assertEqual(cpu_cost, gpu_cost)
        self.assertAlmostEqual(cpu_cost, expected_cost, places)

    def test_empty_label(self):
        probs = torch.FloatTensor([
            [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
            [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]
        ]).contiguous()

        labels = Variable(torch.IntTensor([1, 2]))
        label_sizes = Variable(torch.IntTensor([2, 0]))
        sizes = Variable(torch.IntTensor([2, 2]))
        cpu_cost, gpu_cost = run_grads(label_sizes, labels, probs, sizes)
        expected_cost = 6.416517496109
        self.assertEqual(cpu_cost, gpu_cost)
        self.assertAlmostEqual(cpu_cost, expected_cost, places)


if __name__ == '__main__':
    unittest.main()
