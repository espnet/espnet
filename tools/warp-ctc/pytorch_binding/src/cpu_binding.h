int cpu_ctc(THFloatTensor *probs,
                        THFloatTensor *grads,
                        THIntTensor *labels_ptr,
                        THIntTensor *label_sizes_ptr,
                        THIntTensor *sizes,
                        int minibatch_size,
                        THFloatTensor *costs);